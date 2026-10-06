"""Quantized wrappers around ONNX operators.

Every :class:`BaseQuantLayer` wraps one ONNX node and offers four views:

* :meth:`~BaseQuantLayer.calibrate`: float forward pass that records the
  output ranges,
* :meth:`~BaseQuantLayer.quantized_infer`: fake quantization (Q/DQ
  simulation on float tensors),
* :meth:`~BaseQuantLayer.quantized_infer_integer`: integer-only arithmetic
  with fixed-point requantization (activations are int32 arrays holding
  values in ``[qmin, qmax]``),
* :meth:`~BaseQuantLayer.finalize_calibration`: turns the recorded ranges
  into scales / zero points and precomputes all integer parameters.

All tensor-level quantization parameters live in the owning
:class:`~quantization.quant_container.QuantContainer` (``qparams``), so a
layer simply reads its inputs' parameters and publishes its outputs'.
"""

import numpy as np

from quantization import int_ops, native, pwl
from quantization.float_ops import UNARY_OPS, conv_pads, run_node
from quantization.utils import (
    QParams,
    calculate_scale_zp,
    dequantize,
    fake_quantize,
    parse_quant_type,
    quantize,
    quantize_multiplier,
    quantize_multipliers,
)

# Largest activation lookup table we are willing to build (16-bit tensors).
MAX_LUT_BITS = 16


def _min_max(x, cur_min, cur_max):
    lo, hi = float(np.min(x)), float(np.max(x))
    if cur_min is None:
        return lo, hi
    return min(cur_min, lo), max(cur_max, hi)


def _bias_range(num_bits):
    return (-(2**63), 2**63 - 1) if num_bits > 8 else (-(2**31), 2**31 - 1)


def _weight_qparams(w, axis, quant_type, per_channel):
    """Quantization parameters of a weight tensor.

    Returns:
        ``(scale, zp)``: float64 / int64 arrays with one entry per slice along
        ``axis`` (every entry identical when ``per_channel`` is False).
    """
    flat = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
    if per_channel:
        params = [
            calculate_scale_zp(r.min(), r.max(), quant_type)[:2] for r in flat
        ]
    else:
        sc, zp = calculate_scale_zp(flat.min(), flat.max(), quant_type)[:2]
        params = [(sc, zp)] * flat.shape[0]
    scale = np.array([p[0] for p in params], dtype=np.float64)
    zp = np.array([p[1] for p in params], dtype=np.int64)
    return scale, zp


class BaseQuantLayer:
    """Generic quantized node.

    Works for any operator that has a NumPy implementation by sandwiching it
    between dequantization and quantization. Subclasses replace the integer
    path with real integer arithmetic.
    """

    #: ``"pwl"``: non-linear functions as <= 4-segment piecewise-linear integer
    #: functions (full integer pipeline); ``"lut"``: tabulated / float.
    nonlinear = "pwl"

    def __init__(self, node, graph, quant_type="int8", per_channel=True):
        self.node = node
        self.graph = graph
        self.name = node.name
        self.quant_type = quant_type
        self.per_channel = per_channel
        self.num_bits, self.signed = parse_quant_type(quant_type)
        # Tensors consumed as quantized activations (constants excluded unless
        # a subclass opts in, e.g. ``x + const``).
        self.data_inputs = self._select_data_inputs()
        self.out_min = [None] * len(node.outputs)
        self.out_max = [None] * len(node.outputs)
        self.in_qp = []
        self.out_qp = []
        self.in_range = []

    # -------------------------------------------------------- bookkeeping ----
    def _select_data_inputs(self):
        return [
            i for i in self.node.inputs if i and not self.graph.is_constant(i)
        ]

    def _qp(self, qps, idx=0):
        return qps[idx] if len(qps) > idx else None

    in_scale = property(lambda s: s._qp(s.in_qp).scale if s.in_qp else None)
    in_zp = property(lambda s: s._qp(s.in_qp).zp if s.in_qp else None)
    in_qmin = property(lambda s: s._qp(s.in_qp).qmin if s.in_qp else None)
    in_qmax = property(lambda s: s._qp(s.in_qp).qmax if s.in_qp else None)
    out_scale = property(lambda s: s._qp(s.out_qp).scale if s.out_qp else None)
    out_zp = property(lambda s: s._qp(s.out_qp).zp if s.out_qp else None)
    out_qmin = property(lambda s: s._qp(s.out_qp).qmin if s.out_qp else None)
    out_qmax = property(lambda s: s._qp(s.out_qp).qmax if s.out_qp else None)

    # ------------------------------------------------------------ float ----
    def _full_args(self, data_args):
        """Merges runtime data tensors with the node's constant inputs."""
        data = dict(zip(self.data_inputs, data_args))
        args = []
        for name in self.node.inputs:
            if not name:
                args.append(None)
            elif name in data:
                args.append(data[name])
            else:
                args.append(self.graph.initializers[name])
        return args

    def _run_float(self, data_args, **kw):
        return run_node(
            self.node, self._full_args(data_args), self.graph.opset, **kw
        )

    def calibrate(self, inputs):
        """Float forward pass that records the output min / max."""
        outs = self._run_float(inputs)
        for i, o in enumerate(outs):
            self.out_min[i], self.out_max[i] = _min_max(
                o, self.out_min[i], self.out_max[i]
            )
        return outs

    # -------------------------------------------------------- finalize ----
    def _default_out_qparams(self):
        qps = []
        for i, name in enumerate(self.node.outputs):
            if self.out_min[i] is None:
                raise ValueError(
                    f"Layer {self.name}: output '{name}' was never observed; "
                    "run calibrate() before finalizing"
                )
            qps.append(
                QParams(
                    *calculate_scale_zp(
                        self.out_min[i], self.out_max[i], self.quant_type
                    )
                )
            )
        return qps

    def finalize_calibration(self, container):
        """Derives all quantization parameters from the observed ranges."""
        self.in_qp = [container.qparams[n] for n in self.data_inputs]
        self.in_range = [
            container.tensor_range.get(n) for n in self.data_inputs
        ]
        self.out_qp = self._default_out_qparams()
        for i, (name, qp) in enumerate(zip(self.node.outputs, self.out_qp)):
            container.qparams[name] = qp
            if self.out_min[i] is not None:
                container.tensor_range[name] = (
                    self.out_min[i],
                    self.out_max[i],
                )
        self.prepare()

    def load_qparams(self, container):
        """Restores the parameters from ``container.qparams`` (after load)."""
        self.in_qp = [container.qparams[n] for n in self.data_inputs]
        self.in_range = [
            container.tensor_range.get(n) for n in self.data_inputs
        ]
        self.out_qp = [container.qparams[n] for n in self.node.outputs]
        self.prepare()

    def prepare(self):
        """Precomputes integer parameters once in/out qparams are known."""

    def _check_ready(self):
        if not self.out_qp or (self.data_inputs and not self.in_qp):
            raise ValueError(f"Layer {self.name} is not calibrated.")

    # ------------------------------------------------- quantized paths ----
    @staticmethod
    def _fq(x, qp):
        return fake_quantize(x, qp.scale, qp.zp, qp.qmin, qp.qmax)

    def _fq_inputs(self, inputs):
        return [self._fq(x, qp) for x, qp in zip(inputs, self.in_qp)]

    def _fq_outputs(self, outs):
        return [self._fq(o, qp) for o, qp in zip(outs, self.out_qp)]

    def quantized_infer(self, inputs):
        """Fake-quantized forward pass on float tensors."""
        self._check_ready()
        outs = self._run_float(self._fq_inputs(inputs))
        return self._fq_outputs(outs)

    def quantized_infer_integer(self, inputs):
        """Integer pass (fallback: dequantize -> float op -> quantize)."""
        self._check_ready()
        x = [
            dequantize(v, qp.scale, qp.zp) for v, qp in zip(inputs, self.in_qp)
        ]
        outs = self._run_float(x)
        return [
            quantize(o, qp.scale, qp.zp, qp.qmin, qp.qmax)
            for o, qp in zip(outs, self.out_qp)
        ]


# ------------------------------------------------------------------- linear --
class _WeightedLayer(BaseQuantLayer):
    """Shared weight / bias quantization for Conv and Gemm style layers."""

    # axis of the weight tensor holding the output channels
    weight_axis = 0

    def __init__(self, node, graph, quant_type="int8", per_channel=True):
        super().__init__(node, graph, quant_type, per_channel)
        self.w_qmin, self.w_qmax = (
            (-(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1)
            if self.signed
            else (0, 2**self.num_bits - 1)
        )
        self.b_qmin, self.b_qmax = _bias_range(self.num_bits)
        self.b_zp = 0
        self.w_scale = None
        self.w_zp = None
        self.b_scale = None
        self.bias_int = None

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    def _weights(self):
        """Float ``(weight, bias or None)`` laid out for this layer."""
        raise NotImplementedError

    def _channel_shape(self, ndim):
        shape = [1] * ndim
        shape[self.weight_axis] = -1
        return shape

    def prepare(self):
        w, bias = self._weights()
        in_scale = self.in_scale
        scale_c, zp_c = _weight_qparams(
            w, self.weight_axis, self.quant_type, self.per_channel
        )
        bshape = self._channel_shape(w.ndim)
        self._w_scale_c, self._w_zp_c = scale_c, zp_c
        if self.per_channel:
            self.w_scale, self.w_zp = scale_c, zp_c
        else:
            self.w_scale, self.w_zp = float(scale_c[0]), int(zp_c[0])

        self._reduction = int(np.prod(w.shape) // w.shape[self.weight_axis])
        self.wide = native.accumulator_is_wide(self.num_bits, self._reduction)
        self.w_q = quantize(
            w,
            scale_c.reshape(bshape),
            zp_c.reshape(bshape),
            self.w_qmin,
            self.w_qmax,
        )
        # int16 operands on the narrow path, int32 on the wide (int64) path.
        self.w_centered = (
            self.w_q.astype(np.int64) - zp_c.reshape(bshape)
        ).astype(np.int32 if self.wide else np.int16)
        self.fake_w = dequantize(
            self.w_q, scale_c.reshape(bshape), zp_c.reshape(bshape)
        )

        self.b_scale = in_scale * self.w_scale
        if bias is not None:
            b_scale_c = in_scale * scale_c
            self.bias_int = np.clip(
                np.round(bias / b_scale_c), self.b_qmin, self.b_qmax
            ).astype(np.int64)
            self.fake_bias = (self.bias_int * b_scale_c).astype(np.float32)
        else:
            self.bias_int = None
            self.fake_bias = None

        real = in_scale * scale_c / self.out_scale
        self.q_mult, self.shift = quantize_multipliers(real)
        if not self.per_channel:
            # Per-tensor weights: scalar multiplier / shift like a classic
            # TFLite per-tensor layer.
            self.q_mult, self.shift = self.q_mult[:1], self.shift[:1]


class ConvQuantize(_WeightedLayer):
    """2-D convolution (grouped / depthwise) with constant weights."""

    def _weights(self):
        w = self.graph.initializers[self.node.inputs[1]].astype(np.float32)
        bias = None
        if len(self.node.inputs) > 2 and self.node.inputs[2]:
            bias = self.graph.initializers[self.node.inputs[2]].astype(
                np.float32
            )
        return w, bias

    def _geometry(self, x_shape):
        w_shape = self.w_q.shape
        strides = tuple(self.node.attr("strides", [1, 1]))
        dilations = tuple(self.node.attr("dilations", [1, 1]))
        pads = conv_pads(
            self.node, x_shape[2:], w_shape[2:], strides, dilations
        )
        return strides, pads, dilations, self.node.attr("group", 1)

    def quantized_infer(self, inputs):
        self._check_ready()
        x = self._fq_inputs(inputs)[0]
        strides, pads, dil, groups = self._geometry(x.shape)
        y = native.conv2d_f32(x, self.fake_w, strides, pads, dil, groups)
        if self.fake_bias is not None:
            y = y + self.fake_bias.reshape(1, -1, 1, 1)
        return self._fq_outputs([y])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        x = np.asarray(inputs[0])
        strides, pads, dil, groups = self._geometry(x.shape)
        x_centered = x - self.in_zp
        acc = native.conv2d_int(
            x_centered, self.w_centered, strides, pads, dil, groups, self.wide
        )
        n, cout, oh, ow = acc.shape
        out = native.requantize(
            acc,
            n,
            cout,
            oh * ow,
            self.q_mult,
            self.shift,
            self.out_zp,
            self.out_qmin,
            self.out_qmax,
            pre_bias=self.bias_int,
        )
        return [out]


class GemmQuantize(_WeightedLayer):
    """``Gemm`` / ``MatMul`` against a constant weight matrix."""

    weight_axis = 1  # weights are laid out [K, N]

    def _weights(self):
        node = self.node
        b = self.graph.initializers[node.inputs[1]].astype(np.float32)
        bias = None
        if node.op_type == "Gemm":
            if node.attr("transB", 0):
                b = b.T
            b = b * np.float32(node.attr("alpha", 1.0))
            if len(node.inputs) > 2 and node.inputs[2]:
                c = self.graph.initializers[node.inputs[2]].astype(np.float32)
                bias = np.broadcast_to(c.reshape(-1), (b.shape[1],)).astype(
                    np.float32
                )
                bias = bias * np.float32(node.attr("beta", 1.0))
        elif len(node.inputs) > 2 and node.inputs[2]:  # fused MatMul bias
            bias = self.graph.initializers[node.inputs[2]].astype(np.float32)
            bias = np.ascontiguousarray(bias.reshape(-1))
        return np.ascontiguousarray(b), bias

    def _flatten(self, x):
        return x.reshape(-1, x.shape[-1])

    def _out_shape(self, x):
        if self.node.op_type == "Gemm":
            return (x.shape[0], self.w_q.shape[1])
        return x.shape[:-1] + (self.w_q.shape[1],)

    def quantized_infer(self, inputs):
        self._check_ready()
        x = self._fq_inputs(inputs)[0]
        y = native.gemm_f32(self._flatten(x), self.fake_w)
        if self.fake_bias is not None:
            y = y + self.fake_bias
        return self._fq_outputs([y.reshape(self._out_shape(x))])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        x = np.asarray(inputs[0])
        x2 = self._flatten(x) - self.in_zp
        acc = native.gemm_int(x2, self.w_centered, self.wide)
        m, n = acc.shape
        out = native.requantize(
            acc,
            m,
            n,
            1,
            self.q_mult,
            self.shift,
            self.out_zp,
            self.out_qmin,
            self.out_qmax,
            pre_bias=self.bias_int,
        )
        return [out.reshape(self._out_shape(x))]


class MultiHeadAttentionQuantize(BaseQuantLayer):
    """Attention core with integer ``q k^T`` and ``p v`` matrix products.

    ``q``, ``k`` and ``v`` come from one QKV tensor, so they share its scale
    and zero point. ``q k^T`` is accumulated in integers and dequantized with
    ``s_x ** 2``; the softmax runs in float on that (small) tensor and its
    result is quantized with a *fixed* probability scale ``1 / (qmax - qmin)``
    (the range ``[0, 1]`` is known, no calibration needed); ``p v`` is again
    an integer product requantized to the output grid.
    """

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    def prepare(self):
        qp, out = self.in_qp[0], self.out_qp[0]
        self.p_scale = 1.0 / (qp.qmax - qp.qmin)
        self.p_zp = qp.qmin
        self.p_qmin, self.p_qmax = qp.qmin, qp.qmax
        self.q_mult, self.shift = quantize_multiplier(
            self.p_scale * qp.scale / out.scale
        )

    def _split_heads(self, x):
        heads, hd = self.node.attr("num_heads"), self.node.attr("head_dim")
        b, t = x.shape[:2]
        return x.reshape(b, t, 3, heads, hd).transpose(2, 0, 3, 1, 4)

    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def quantized_infer(self, inputs):
        self._check_ready()
        x = self._fq_inputs(inputs)[0]
        q, k, v = self._split_heads(x)
        logits = np.matmul(q, k.transpose(0, 1, 3, 2)).astype(np.float64)
        soft = (
            int_ops.softmax_float(logits)
            if self.nonlinear == "pwl"
            else self._softmax(logits)
        )
        probs = fake_quantize(
            soft,
            self.p_scale,
            self.p_zp,
            self.p_qmin,
            self.p_qmax,
        )
        o = np.matmul(probs.astype(np.float32), v)
        b, t = x.shape[:2]
        o = o.transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self._fq_outputs([o])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        qp, out = self.in_qp[0], self.out_qp[0]
        x = np.asarray(inputs[0])
        b, t = x.shape[:2]
        q, k, v = self._split_heads((x - qp.zp).astype(np.int32))
        heads, hd = q.shape[1], q.shape[3]
        wide_qk = native.accumulator_is_wide(self.num_bits, hd)
        wide_pv = native.accumulator_is_wide(self.num_bits, t)
        acc_dtype = np.int64 if wide_pv else np.int32
        o_acc = np.empty((b, heads, t, hd), dtype=acc_dtype)
        for i in range(b):
            for h in range(heads):
                acc = native.gemm_int(
                    q[i, h], np.ascontiguousarray(k[i, h].T), wide_qk
                )
                if self.nonlinear == "pwl":  # integer exp / reciprocal
                    p_int = int_ops.softmax(
                        acc,
                        qp.scale * qp.scale,
                        self.p_scale,
                        self.p_zp,
                        self.p_qmin,
                        self.p_qmax,
                    )
                else:
                    p = self._softmax(
                        acc.astype(np.float64) * (qp.scale * qp.scale)
                    )
                    p_int = np.clip(
                        np.round(p / self.p_scale) + self.p_zp,
                        self.p_qmin,
                        self.p_qmax,
                    ).astype(np.int32)
                o_acc[i, h] = native.gemm_int(
                    p_int - self.p_zp, v[i, h], wide_pv
                )
        merged = np.ascontiguousarray(o_acc.transpose(0, 2, 1, 3))
        res = native.requantize(
            merged.reshape(1, 1, -1),
            1,
            1,
            merged.size,
            self.q_mult,
            self.shift,
            out.zp,
            out.qmin,
            out.qmax,
        )
        return [res.reshape(b, t, heads * hd)]


class BatchNormalizationQuantize(BaseQuantLayer):
    """Inference-mode BatchNorm as a per-channel integer affine transform."""

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    def prepare(self):
        gamma, beta, mean, var = (
            self.graph.initializers[n].astype(np.float64)
            for n in self.node.inputs[1:5]
        )
        eps = self.node.attr("epsilon", 1e-5)
        self.A = gamma / np.sqrt(var + eps)
        self.B = beta - self.A * mean
        self.q_mult, self.shift = quantize_multipliers(
            self.in_scale * self.A / self.out_scale
        )
        self.q_B = np.round(self.B / self.out_scale).astype(np.int64)

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        x = np.asarray(inputs[0])
        centered = (x - self.in_zp).astype(np.int32)
        n, c = x.shape[0], x.shape[1]
        out = native.requantize(
            centered,
            n,
            c,
            int(np.prod(x.shape[2:], dtype=np.int64)),
            self.q_mult,
            self.shift,
            self.out_zp,
            self.out_qmin,
            self.out_qmax,
            post_bias=self.q_B,
        )
        return [out.reshape(x.shape)]


# -------------------------------------------------------------- activations --
class ActivationQuantize(BaseQuantLayer):
    """Element-wise unary function as an integer piecewise-linear function.

    With ``nonlinear="pwl"`` (default) the function is approximated on the
    calibrated input range by a continuous PWL with at most four segments and
    evaluated with integer arithmetic only (one compare per knot, one
    fixed-point multiply). With ``nonlinear="lut"`` it is tabulated instead
    (inputs of at most 16 bits).
    """

    pwl_domain_margin = 0.01
    #: ``"data"`` (default) fits the expected squared error under the
    #: calibration distribution; ``"minimax"`` the worst case over the
    #: whole range.
    pwl_fit = "data"
    #: percentage of samples ignored at each end when choosing the fit domain
    #: (the end segments extend linearly beyond it); 0 keeps the full range.
    pwl_clip_percentile = 0.05
    max_samples = 200_000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._samples = []
        self._rng = np.random.default_rng(0)
        self.pwl_state_override = None

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    def calibrate(self, inputs):
        flat = np.asarray(inputs[0]).reshape(-1)
        keep = min(flat.size, 20_000)
        picked = self._rng.choice(flat, keep, replace=False)
        self._samples.append(
            np.concatenate([picked, [flat.min(), flat.max()]])
        )
        return super().calibrate(inputs)

    @staticmethod
    def supports(node, graph, quant_type, nonlinear="pwl"):
        """True if ``node`` can be evaluated for ``quant_type``."""
        num_bits, _ = parse_quant_type(quant_type)
        if nonlinear == "lut" and num_bits > MAX_LUT_BITS:
            return False
        if node.op_type == "Clip":
            return all(graph.is_constant(i) for i in node.inputs[1:] if i)
        return node.op_type in UNARY_OPS or node.op_type == "FusedElementwise"

    def _real_function(self, x):
        return self._run_float([np.asarray(x, dtype=np.float32)])[0]

    def prepare(self):
        qp, o = self.in_qp[0], self.out_qp[0]
        if self.nonlinear == "lut":
            q_in = np.arange(qp.qmin, qp.qmax + 1, dtype=np.int64)
            x = dequantize(q_in.astype(np.int32), qp.scale, qp.zp)
            y = self._run_float([x])[0]
            self.lut = quantize(y, o.scale, o.zp, o.qmin, o.qmax).astype(
                np.int32
            )
            self.lut_offset = qp.qmin
            return
        lo, hi = self.in_range[0] or (
            (qp.qmin - qp.zp) * qp.scale,
            (qp.qmax - qp.zp) * qp.scale,
        )
        pad = self.pwl_domain_margin * max(hi - lo, qp.scale)
        lo = max(lo - pad, (qp.qmin - qp.zp) * qp.scale)
        hi = min(hi + pad, (qp.qmax - qp.zp) * qp.scale)
        samples = (
            np.concatenate(self._samples)[: self.max_samples * 2]
            if self._samples
            else None
        )
        if self.pwl_state_override is not None:  # restored by load()
            x, y = self.pwl_state_override
            self.pwl_float = pwl.PWL(np.asarray(x), np.asarray(y))
        else:
            fit_lo, fit_hi = lo, hi
            if samples is not None and self.pwl_clip_percentile > 0:
                p = self.pwl_clip_percentile
                fit_lo, fit_hi = np.percentile(samples, [p, 100 - p])
                fit_lo, fit_hi = max(fit_lo, lo), min(fit_hi, hi)
            self.pwl_float = pwl.fit_pwl(
                self._real_function,
                fit_lo,
                fit_hi,
                samples=samples if self.pwl_fit == "data" else None,
            )
        self.pwl_int = pwl.to_integer(
            self.pwl_float, qp.scale, qp.zp, o.scale, o.zp, o.qmin, o.qmax
        )
        self.pwl_domain = (lo, hi)
        self.pwl_error = self.pwl_float.max_error
        self.pwl_rmse = None
        if samples is not None:
            err = self.pwl_float(samples) - self._real_function(samples)
            self.pwl_rmse = float(np.sqrt(np.mean(err**2)))

    def quantized_infer(self, inputs):
        if self.nonlinear == "lut":
            return super().quantized_infer(inputs)
        self._check_ready()
        x = self._fq_inputs(inputs)[0]
        y = self.pwl_float(x).astype(np.float32)
        return self._fq_outputs([y])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        x = np.asarray(inputs[0])
        if self.nonlinear == "lut":
            return [native.lut(x, self.lut, self.lut_offset)]
        return [self.pwl_int(x)]


class SoftmaxQuantize(BaseQuantLayer):
    """Integer-only softmax (``2^x`` by shifts + PWL, reciprocal by PWL)."""

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    @staticmethod
    def supports(node, graph, opset):
        """Softmax over a constant axis (opset >= 13 semantics)."""
        return node.op_type == "Softmax" and (
            opset >= 13 or node.attr("axis", -1) == -1
        )

    def quantized_infer(self, inputs):
        self._check_ready()
        x = self._fq_inputs(inputs)[0]
        p = int_ops.softmax_float(x, self.node.attr("axis", -1))
        return self._fq_outputs([p.astype(np.float32)])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        qp, o = self.in_qp[0], self.out_qp[0]
        x = np.asarray(inputs[0])
        axis = self.node.attr("axis", -1)
        moved = np.moveaxis(x, axis, -1)
        codes = int_ops.softmax(moved, qp.scale, o.scale, o.zp, o.qmin, o.qmax)
        return [np.ascontiguousarray(np.moveaxis(codes, -1, axis))]


class LayerNormQuantize(BaseQuantLayer):
    """Integer-only LayerNorm over the last axis (``1/sqrt`` by PWL)."""

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    @staticmethod
    def supports(node, graph):
        """LayerNorm over the last axis with constant gamma / beta."""
        if node.op_type != "LayerNormalization":
            return False
        if node.attr("axis", -1) != -1:
            return False
        return all(graph.is_constant(i) for i in node.inputs[1:] if i)

    def prepare(self):
        init = self.graph.initializers
        gamma = init[self.node.inputs[1]]
        beta = (
            init[self.node.inputs[2]]
            if len(self.node.inputs) > 2 and self.node.inputs[2]
            else np.zeros_like(gamma)
        )
        o = self.out_qp[0]
        self.eps = float(self.node.attr("epsilon", 1e-5))
        self._mult, self._shift, self._post = int_ops.prepare_layer_norm(
            gamma, beta, o.scale
        )

    def quantized_infer(self, inputs):
        self._check_ready()
        init = self.graph.initializers
        x = self._fq_inputs(inputs)[0]
        gamma = init[self.node.inputs[1]]
        beta = (
            init[self.node.inputs[2]]
            if len(self.node.inputs) > 2 and self.node.inputs[2]
            else 0.0
        )
        y = int_ops.layer_norm_float(x, gamma, beta, self.eps)
        return self._fq_outputs([y.astype(np.float32)])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        qp, o = self.in_qp[0], self.out_qp[0]
        out = int_ops.layer_norm(
            np.asarray(inputs[0]),
            qp.scale,
            self.eps,
            self._mult,
            self._shift,
            self._post,
            o.zp,
            o.qmin,
            o.qmax,
        )
        return [out]


# ------------------------------------------------------------- element-wise --
class _MultiInputLayer(BaseQuantLayer):
    """Layers whose constant operands are quantized like activations."""

    def _select_data_inputs(self):
        return [i for i in self.node.inputs if i]

    def _mults(self, in_scales):
        return [quantize_multiplier(s / self.out_scale) for s in in_scales]


class AddQuantize(_MultiInputLayer):
    """``Add`` / ``Sub``: both operands are rescaled to the output scale."""

    def prepare(self):
        self._mult = self._mults([qp.scale for qp in self.in_qp])
        self._sign = -1 if self.node.op_type == "Sub" else 1

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        (ma, sa), (mb, sb) = self._mult
        a, b = self.in_qp
        o = self.out_qp[0]
        return [
            native.add(
                inputs[0],
                inputs[1],
                a.zp,
                ma,
                sa,
                b.zp,
                mb,
                sb,
                o.zp,
                o.qmin,
                o.qmax,
                self._sign,
            )
        ]


class MulQuantize(_MultiInputLayer):
    """Element-wise product ``(q1 - z1) * (q2 - z2)`` with one requantize."""

    def prepare(self):
        self._mult, self._shift = quantize_multiplier(
            self.in_qp[0].scale * self.in_qp[1].scale / self.out_scale
        )

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        a, b = self.in_qp
        o = self.out_qp[0]
        return [
            native.mul(
                inputs[0],
                inputs[1],
                a.zp,
                b.zp,
                self._mult,
                self._shift,
                o.zp,
                o.qmin,
                o.qmax,
            )
        ]


class ConcatQuantize(_MultiInputLayer):
    """Rescales every input to the output grid, then concatenates."""

    def prepare(self):
        self._mult = self._mults([qp.scale for qp in self.in_qp])

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        o = self.out_qp[0]
        parts = []
        for x, qp, (m, s) in zip(inputs, self.in_qp, self._mult):
            if qp.scale == o.scale and qp.zp == o.zp:
                parts.append(np.clip(x, o.qmin, o.qmax).astype(np.int32))
            else:
                parts.append(
                    native.rescale(x, qp.zp, m, s, o.zp, o.qmin, o.qmax)
                )
        return [np.concatenate(parts, axis=self.node.attr("axis"))]


class GlobalAveragePoolQuantize(BaseQuantLayer):
    """Spatial mean: int64 sum with ``1/area`` folded into the scale."""

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        x = np.asarray(inputs[0])
        axes = tuple(range(2, x.ndim))
        area = int(np.prod(x.shape[2:]))
        acc = (x.astype(np.int64) - self.in_zp).sum(axis=axes, keepdims=True)
        mult, shift = quantize_multiplier(
            self.in_scale / (self.out_scale * area)
        )
        n, c = x.shape[:2]
        out = native.requantize(
            acc,
            n,
            c,
            1,
            mult,
            shift,
            self.out_zp,
            self.out_qmin,
            self.out_qmax,
        )
        return [out]


class ShapingQuantize(BaseQuantLayer):
    """Scale-preserving layout / selection operators.

    Output parameters are copied from the input, so the integer path simply
    applies the operator to the integer tensor (padding with the zero point,
    max-pooling with ``qmin``).
    """

    SUPPORTED = {
        "Gather",
        "Reshape",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
        "Split",
        "Slice",
        "Pad",
        "MaxPool",
        "GlobalMaxPool",
    }

    def _select_data_inputs(self):
        return [self.node.inputs[0]]

    @classmethod
    def supports(cls, node, graph):
        """True if ``node`` is scale preserving and its extras are constant."""
        if node.op_type not in cls.SUPPORTED:
            return False
        if not all(graph.is_constant(i) for i in node.inputs[1:] if i):
            return False
        if node.op_type == "Pad":
            if node.attr("mode", "constant") != "constant":
                return False
            value = (
                graph.initializers[node.inputs[2]]
                if len(node.inputs) > 2 and node.inputs[2]
                else node.attr("value", 0)
            )
            return float(np.asarray(value)) == 0.0
        return True

    def _default_out_qparams(self):
        return [self.in_qp[0]] * len(self.node.outputs)

    def quantized_infer(self, inputs):
        self._check_ready()
        return self._run_float(self._fq_inputs(inputs))

    def quantized_infer_integer(self, inputs):
        self._check_ready()
        kw = {}
        if self.node.op_type == "Pad":
            kw["pad_value"] = self.in_zp
        elif self.node.op_type == "MaxPool":
            kw["pad_value"] = self.in_qmin
        outs = self._run_float([np.asarray(inputs[0])], **kw)
        return [np.ascontiguousarray(o, dtype=np.int32) for o in outs]


def map_node_to_quant(
    node, graph, quant_type="int8", per_channel=True, nonlinear="pwl"
):
    """Chooses the quantized layer implementation for an ONNX node.

    ``nonlinear="pwl"`` evaluates every non-linear function (activations,
    softmax, layer norm) with integer piecewise-linear functions of at most
    four segments; ``"lut"`` keeps lookup tables and float fallbacks.
    """
    if nonlinear not in ("pwl", "lut"):
        raise ValueError("nonlinear must be 'pwl' or 'lut'")
    layer = _map_node(node, graph, quant_type, per_channel, nonlinear)
    layer.nonlinear = nonlinear
    return layer


def _map_node(node, graph, quant_type, per_channel, nonlinear):
    op = node.op_type
    const = graph.is_constant
    args = (node, graph, quant_type, per_channel)

    if op == "Conv" and len(node.inputs) > 1 and const(node.inputs[1]):
        w = graph.initializers[node.inputs[1]]
        bias_ok = (
            len(node.inputs) < 3 or not node.inputs[2] or const(node.inputs[2])
        )
        if w.ndim == 4 and bias_ok:
            return ConvQuantize(*args)
    if (
        op in ("Gemm", "MatMul", "LinearLayer")
        and len(node.inputs) > 1
        and const(node.inputs[1])
    ):
        b = graph.initializers[node.inputs[1]]
        ok = b.ndim == 2 and not (op == "Gemm" and node.attr("transA", 0))
        if len(node.inputs) > 2 and node.inputs[2]:
            ok = ok and const(node.inputs[2])
        if ok and not const(node.inputs[0]):
            return GemmQuantize(*args)
    if op == "MultiHeadAttention" and not const(node.inputs[0]):
        return MultiHeadAttentionQuantize(*args)
    if op == "BatchNormalization" and all(const(i) for i in node.inputs[1:5]):
        return BatchNormalizationQuantize(*args)
    if nonlinear == "pwl" and SoftmaxQuantize.supports(
        node, graph, graph.opset
    ):
        return SoftmaxQuantize(*args)
    if nonlinear == "pwl" and LayerNormQuantize.supports(node, graph):
        return LayerNormQuantize(*args)
    if ActivationQuantize.supports(node, graph, quant_type, nonlinear):
        return ActivationQuantize(*args)
    if op in ("Add", "Sub") and not all(const(i) for i in node.inputs):
        return AddQuantize(*args)
    if op == "Mul" and not all(const(i) for i in node.inputs):
        return MulQuantize(*args)
    if op == "Concat":
        return ConcatQuantize(*args)
    if op == "GlobalAveragePool":
        return GlobalAveragePoolQuantize(*args)
    if ShapingQuantize.supports(node, graph):
        return ShapingQuantize(*args)
    return BaseQuantLayer(*args)
