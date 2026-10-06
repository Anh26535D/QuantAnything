"""Quantization container: float / fake-quant / integer ONNX execution."""

import json
from pathlib import Path

import numpy as np

from quantization import float_ops
from quantization.onnx_graph import OnnxGraph
from quantization.quant_layers import map_node_to_quant
from quantization.utils import (
    QParams,
    calculate_scale_zp,
    dequantize,
    fake_quantize,
    quantize,
)

MODES = ("float", "calibrate", "quantize", "integer")


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

    def __init__(self, model, quant_type="int8", per_channel=True):
        self.graph = OnnxGraph.from_model(model)
        self.quant_type = quant_type
        self.per_channel = per_channel
        self.quant_layers = {}
        for node in self.graph.nodes:
            self.quant_layers[node.name] = map_node_to_quant(
                node, self.graph, quant_type, per_channel
            )
        self.qparams = {}
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
        for name, (lo, hi) in self._input_range.items():
            self.qparams[name] = QParams(
                *calculate_scale_zp(lo, hi, self.quant_type)
            )
        self._const_q = {}
        for name in self._const_names:
            value = self.graph.initializers[name].astype(np.float32)
            qp = QParams(
                *calculate_scale_zp(value.min(), value.max(), self.quant_type)
            )
            self.qparams[name] = qp
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
            "qparams": {k: v.to_dict() for k, v in self.qparams.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, model, path):
        """Rebuilds a calibrated container from ``model`` and a saved JSON."""
        payload = json.loads(Path(path).read_text())
        container = cls(model, payload["quant_type"], payload["per_channel"])
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
        for layer in container.quant_layers.values():
            layer.load_qparams(container)
        container.calibrated = True
        return container
