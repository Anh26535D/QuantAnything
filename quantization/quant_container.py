"""Quantization container: float / fake-quant / integer ONNX execution."""

import json
from pathlib import Path

import numpy as np

from quantization import float_ops, graph_passes
from quantization.onnx_graph import OnnxGraph
from quantization.quant_layers import BaseQuantLayer, map_node_to_quant
from quantization.utils import (
    QParams,
    calculate_scale_zp,
    dequantize,
    fake_quantize,
    quantize,
)

MODES = ("float", "calibrate", "quantize", "integer")


class NotIntegerError(ValueError):
    """Raised in ``strict`` mode when a layer would fall back to float."""


def _is_batches(x):
    """True when ``x`` is an iterable of batches rather than one batch."""
    return not isinstance(x, (np.ndarray, dict))


class QuantContainer:
    """Post-training static quantization of an ONNX model.

    The model is held as a topologically ordered list of quantized layers
    (one per ONNX node). Tensor-level quantization parameters are stored in
    :attr:`qparams`, keyed by tensor name.

    Args:
        model: ``onnx.ModelProto``, path to an ``.onnx`` file or an
            :class:`~quantization.onnx_graph.OnnxGraph`.
        quant_type: e.g. ``"int8"``, ``"uint8"``, ``"int16"``.
        per_channel: quantize conv / dense weights per output channel.
    """

    def __init__(
        self,
        model,
        quant_type="int8",
        per_channel=True,
        nonlinear="pwl",
        strict=False,
        precision=None,
        residual=None,
        residual_bits=16,
        fuse_residual=True,
        add_impl="exact",
    ):
        if residual not in (None, "accumulate", "lazy"):
            raise ValueError("residual must be None, 'accumulate' or 'lazy'")
        self.graph = OnnxGraph.from_model(model)
        self.fused_residuals = 0
        if residual == "accumulate" and fuse_residual:
            self.fused_residuals = graph_passes.fuse_residual_add(self.graph)
        self.quant_type = quant_type
        self.per_channel = per_channel
        self.nonlinear = nonlinear
        self.quant_layers = {}
        for node in self.graph.nodes:
            self.quant_layers[node.name] = map_node_to_quant(
                node, self.graph, quant_type, per_channel, nonlinear
            )
        for layer in self.quant_layers.values():
            if hasattr(layer, "add_impl"):
                layer.add_impl = add_impl
        self.residual = residual
        self.residual_bits = residual_bits
        self.precision = dict(precision or {})
        for layer in self.quant_layers.values():  # mixed precision
            override = self.precision.get(layer.name) or self.precision.get(
                layer.node.op_type
            )
            if override:
                layer.out_quant_type = override
        fallbacks = self.float_fallbacks()
        if strict and fallbacks:
            raise NotIntegerError(
                "layers without an integer implementation: "
                + ", ".join(f"{l.name} ({l.node.op_type})" for l in fallbacks)
            )
        self.qparams = {}
        self.tensor_range = {}
        self._input_range = {}
        self._const_names = self._constant_operands()
        self._const_q = {}
        self.calibrated = False

    # ------------------------------------------------------------ helpers ----
    def _constant_operands(self):
        """Constants consumed as quantized activations (e.g. ``x + bias``)."""
        names = []
        for layer in self.quant_layers.values():
            for n in layer.data_inputs:
                if self.graph.is_constant(n) and n not in names:
                    names.append(n)
        return names

    def residual_groups(self):
        """Chains of ``Add`` layers that carry a residual stream.

        Two ``Add`` nodes belong to the same stream when the output of one is
        an input of the other (e.g. every skip connection of a transformer).
        Streams with a single ``Add`` are ignored.
        """
        adds = [
            l
            for l in self.layers
            if (
                l.node.op_type == "Add"
                and type(l).__name__ == "AddQuantize"
                and any(not self.graph.is_constant(i) for i in l.node.inputs)
            )
            or (
                type(l).__name__ == "GemmQuantize"
                and getattr(l, "has_residual", False)
            )
        ]
        parent = {l.name: l.name for l in adds}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by_output = {l.node.outputs[0]: l for l in adds}
        for l in adds:
            for tensor in l.node.inputs:
                if tensor in by_output:
                    parent[find(l.name)] = find(by_output[tensor].name)
        groups = {}
        for l in adds:
            groups.setdefault(find(l.name), []).append(l)
        return [g for g in groups.values() if len(g) >= 2]

    def _term_scale(self, tensor):
        """Quantization step of an unmerged residual term (calibrated)."""
        producer = self.graph.producers.get(tensor)
        if producer is not None:
            layer = self.quant_layers[producer.name]
            idx = producer.outputs.index(tensor)
            lo, hi = layer.out_min[idx], layer.out_max[idx]
        elif tensor in self.graph.initializers:
            value = self.graph.initializers[tensor]
            lo, hi = float(value.min()), float(value.max())
        else:
            lo, hi = self._input_range[tensor]
        return calculate_scale_zp(lo, hi, self.quant_type)[0]

    def _stream_operands(self, layer):
        """Tensors a residual-stream layer adds (the stream-side inputs)."""
        if type(layer).__name__ == "AddQuantize":
            return list(layer.data_inputs)
        return [layer.data_inputs[1]]  # fused: x is not a stream input

    def _align_roots(self, tensor, qp, forced, consts):
        """Puts the producer chain of a stream *root* on the stream grid.

        Scale-preserving layers are looked through, a ``Concat`` is forced
        together with its inputs, constants are quantized on the stream grid;
        any other producer simply gets the grid as its output grid.
        """
        if self.graph.is_constant(tensor):
            consts.add(tensor)
            return
        node = self.graph.producers.get(tensor)
        if node is None:  # graph input: keeps its grid
            return
        layer = self.quant_layers[node.name]
        kind = type(layer).__name__
        if kind == "ShapingQuantize":
            self._align_roots(layer.data_inputs[0], qp, forced, consts)
        elif kind == "ConcatQuantize":
            forced.append(layer)
            for t in layer.data_inputs:
                self._align_roots(t, qp, forced, consts)
        else:
            forced.append(layer)

    def _assign_residual_grids(self):
        """Gives every residual stream one shared output grid.

        ``accumulate``: a fixed ``int{residual_bits}`` grid covering the whole
        stream. Every operand of every add is put on that grid (the producers
        of the stream roots too, e.g. the patch embedding and the position
        embedding), so *no add needs a rescale*: it is a plain integer
        addition of two numbers on the same grid, no dequantization and no
        left shift. ``lazy``: the stream is kept at a grid 16x finer than the
        finest term, i.e. the exact sum of the (int8) terms.
        """
        self._stream_consts = {}
        if self.residual is None:
            return
        for group in self.residual_groups():
            lo = min(l.out_min[0] for l in group)
            hi = max(l.out_max[0] for l in group)
            names = {l.node.outputs[0] for l in group}
            roots = [
                t
                for l in group
                for t in self._stream_operands(l)
                if t not in names
            ]
            if self.residual == "accumulate":
                qp = QParams(
                    *calculate_scale_zp(lo, hi, f"int{self.residual_bits}")
                )
                forced, consts = [], set()
                for t in roots:
                    self._align_roots(t, qp, forced, consts)
                # the grid must also cover the roots' own range
                for layer in forced:
                    lo = min(
                        lo, min(m for m in layer.out_min if m is not None)
                    )
                    hi = max(
                        hi, max(m for m in layer.out_max if m is not None)
                    )
                qp = QParams(
                    *calculate_scale_zp(lo, hi, f"int{self.residual_bits}")
                )
                for layer in forced:
                    layer.forced_out_qp = qp
                self._stream_consts.update({t: qp for t in consts})
            else:
                fine = min(self._term_scale(t) for t in roots) / 16.0
                fine = max(fine, max(abs(lo), abs(hi)) / 2**30)
                qp = QParams(fine, 0, -(2**31) + 1, 2**31 - 1)
            for l in group:
                l.forced_out_qp = qp

    def residual_report(self):
        """Per residual add: does it still need a rescale of an operand?

        Returns ``[(layer name, needs_rescale)]``; with
        ``residual="accumulate"`` every entry is ``False`` once calibrated:
        the add is then a plain integer addition (same scale, same zero
        point) - no dequantization, no multiplier, no left shift.
        """
        out = []
        for group in self.residual_groups():
            for l in group:
                o = l.out_qp[0]
                same = [
                    (self.qparams[t].scale, self.qparams[t].zp)
                    == (o.scale, o.zp)
                    for t in self._stream_operands(l)
                ]
                out.append((l.name, not all(same)))
        return out

    def float_fallbacks(self):
        """Layers that still run as dequantize -> float -> quantize."""
        return [l for l in self.layers if type(l) is BaseQuantLayer]

    def integer_report(self):
        """``[(layer name, op type, implementation, integer-only?)]``."""
        return [
            (
                l.name,
                l.node.op_type,
                type(l).__name__,
                type(l) is not BaseQuantLayer,
            )
            for l in self.layers
        ]

    @property
    def layers(self):
        """Quantized layers in execution order."""
        return list(self.quant_layers.values())

    def _feeds(self, x):
        return float_ops.normalize_feeds(self.graph, x)

    # -------------------------------------------------------- calibration ----
    def calibrate(self, x_calib, verbose=False):
        """Runs calibration batches and finalizes every scale / zero point.

        Args:
            x_calib: one batch (array, or dict keyed by input name) or an
                iterable of batches.
        """
        batches = [x_calib] if not _is_batches(x_calib) else x_calib
        count = 0
        for batch in batches:
            self.run_graph(batch, mode="calibrate")
            count += 1
        if count == 0:
            raise ValueError("no calibration data provided")
        self.finalize()
        if verbose:
            print(f"Calibrated on {count} batch(es).")

    def finalize(self):
        """Computes qparams from the collected ranges (after calibrate)."""
        self.qparams = {}
        self.tensor_range = {}
        for name, (lo, hi) in self._input_range.items():
            self.tensor_range[name] = (lo, hi)
            self.qparams[name] = QParams(
                *calculate_scale_zp(lo, hi, self.quant_type)
            )
        self._const_q = {}
        self._assign_residual_grids()
        for name in self._const_names:
            value = self.graph.initializers[name].astype(np.float32)
            if name in self._stream_consts:  # constant on the stream grid
                qp = self._stream_consts[name]
            else:
                qp = QParams(
                    *calculate_scale_zp(
                        value.min(), value.max(), self.quant_type
                    )
                )
            self.qparams[name] = qp
            self.tensor_range[name] = (float(value.min()), float(value.max()))
            self._const_q[name] = quantize(
                value, qp.scale, qp.zp, qp.qmin, qp.qmax
            )
        for layer in self.quant_layers.values():
            layer.finalize_calibration(self)
        self.calibrated = True

    # ------------------------------------------------------- execution ----
    def quantized_infer(self, x, mode="quantize"):
        """Quantized forward pass; ``mode`` is quantize or integer."""
        return self.run_graph(x, mode=mode)

    def run_graph(self, x, mode="quantize"):
        """Executes the graph layer by layer.

        Args:
            x: input array (single-input models) or dict keyed by input name.
            mode: ``"float"`` (reference), ``"calibrate"``, ``"quantize"``
                (fake quantization) or ``"integer"`` (integer-only math,
                outputs dequantized to float).
        """
        if mode not in MODES:
            raise ValueError(f"Unknown graph mode: {mode}")
        if mode == "float":
            return float_ops.run_float(self.graph, x)
        env, outputs = self._execute(x, mode)
        return outputs

    def _execute(self, x, mode, trace=None):
        graph = self.graph
        if mode != "calibrate" and not self.calibrated:
            raise ValueError("Container is not calibrated; call calibrate().")
        feeds = self._feeds(x)
        env = {}
        for name, value in feeds.items():
            if mode == "calibrate":
                lo, hi = float(value.min()), float(value.max())
                cur = self._input_range.get(name)
                self._input_range[name] = (
                    (lo, hi)
                    if cur is None
                    else (min(cur[0], lo), max(cur[1], hi))
                )
                env[name] = value
            else:
                qp = self.qparams[name]
                env[name] = (
                    fake_quantize if mode == "quantize" else quantize
                )(value, qp.scale, qp.zp, qp.qmin, qp.qmax)
        for name in self._const_names:
            env[name] = (
                self._const_q[name]
                if mode == "integer"
                else graph.initializers[name]
            )

        last_use = {}
        for idx, node in enumerate(graph.nodes):
            for n in node.inputs:
                last_use[n] = idx
        keep = set(graph.outputs) if trace is None else None

        for idx, node in enumerate(graph.nodes):
            layer = self.quant_layers[node.name]
            ins = [env[n] for n in layer.data_inputs]
            if mode == "calibrate":
                outs = layer.calibrate(ins)
            elif mode == "quantize":
                outs = layer.quantized_infer(ins)
            else:
                outs = layer.quantized_infer_integer(ins)
            env.update(zip(node.outputs, outs))
            if trace is not None:
                trace(layer, outs)
            else:  # free tensors that no later layer needs
                for n in set(layer.data_inputs):
                    if last_use.get(n) == idx and n not in keep:
                        env.pop(n, None)

        results = []
        for name in graph.outputs:
            value = env[name]
            if mode == "integer":
                qp = self.qparams[name]
                value = dequantize(value, qp.scale, qp.zp)
            results.append(value)
        return env, (results[0] if len(results) == 1 else results)

    # ---------------------------------------------------------- analysis ----
    def compare_layers(self, x, mode="quantize"):
        """Per-layer error of the quantized pass against the float pass.

        Returns:
            ``{layer name: {"max_diff", "mean_diff", "layer_type"}}``; every
            layer is compared on the (cumulative) outputs of the full float and
            the full quantized run.
        """
        if mode not in ("quantize", "integer"):
            raise ValueError("mode must be 'quantize' or 'integer'")
        float_env = float_ops.run_float(self.graph, x, keep_all=True)
        comparison = {}

        def trace(layer, outs):
            diffs = []
            for name, q in zip(layer.node.outputs, outs):
                if mode == "integer":
                    qp = self.qparams[name]
                    q = dequantize(q, qp.scale, qp.zp)
                diffs.append(np.abs(float_env[name] - q))
            comparison[layer.name] = {
                "max_diff": float(max(d.max() for d in diffs)),
                "mean_diff": float(np.mean([d.mean() for d in diffs])),
                "layer_type": layer.node.op_type,
            }

        self._execute(x, mode, trace=trace)
        return comparison

    # ----------------------------------------------------- persistence ----
    def save(self, path):
        """Writes the quantization parameters (not the weights) as JSON.

        Together with the original ONNX file this fully determines the
        quantized model, see :meth:`load`.
        """
        payload = {
            "quant_type": self.quant_type,
            "per_channel": self.per_channel,
            "nonlinear": self.nonlinear,
            "qparams": {k: v.to_dict() for k, v in self.qparams.items()},
            "ranges": {k: list(v) for k, v in self.tensor_range.items()},
            "pwl": {
                l.name: {
                    "x": l.pwl_float.x.tolist(),
                    "y": l.pwl_float.y.tolist(),
                }
                for l in self.layers
                if getattr(l, "pwl_float", None) is not None
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, model, path):
        """Rebuilds a calibrated container from ``model`` and a saved JSON."""
        payload = json.loads(Path(path).read_text())
        container = cls(
            model,
            payload["quant_type"],
            payload["per_channel"],
            payload.get("nonlinear", "lut"),
        )
        container.tensor_range = {
            k: tuple(v) for k, v in payload.get("ranges", {}).items()
        }
        container.qparams = {
            k: QParams.from_dict(v) for k, v in payload["qparams"].items()
        }
        missing = [
            n
            for n in container.graph.input_names + container._const_names
            if n not in container.qparams
        ]
        if missing:
            raise ValueError(f"saved parameters do not match model: {missing}")
        container._const_q = {}
        for name in container._const_names:
            qp = container.qparams[name]
            container._const_q[name] = quantize(
                container.graph.initializers[name].astype(np.float32),
                qp.scale,
                qp.zp,
                qp.qmin,
                qp.qmax,
            )
        for name, state in payload.get("pwl", {}).items():
            container.quant_layers[name].pwl_state_override = (
                state["x"],
                state["y"],
            )
        for layer in container.quant_layers.values():
            layer.load_qparams(container)
        container.calibrated = True
        return container
