"""NumPy reference implementations of the supported ONNX operators.

The operators are written to be dtype-agnostic where that makes sense
(reshape / slice / pooling ...), so the quantized graph can run the very same
code on integer tensors for the scale-preserving "shaping" layers.

Every operator has the signature ``fn(node, inputs, opset, **kw) -> list`` and
returns one array per node output.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantization import native

OPS = {}
# Element-wise unary operators that can be tabulated into a lookup table.
UNARY_OPS = {}


def register(*names):
    """Registers an operator implementation under one or more ONNX names."""

    def deco(fn):
        for name in names:
            OPS[name] = fn
        return fn

    return deco


def unary(name):
    """Registers an element-wise unary operator (LUT-able when quantized)."""

    def deco(fn):
        UNARY_OPS[name] = fn

        def op(node, inputs, opset, **kw):
            return [fn(node, inputs[0])]

        OPS[name] = op
        return fn

    return deco


class UnsupportedOp(NotImplementedError):
    """Raised for ONNX operators without a NumPy implementation."""


def run_node(node, inputs, opset=13, **kw):
    """Executes one node on NumPy inputs and returns its outputs."""
    fn = OPS.get(node.op_type)
    if fn is None:
        raise UnsupportedOp(
            f"ONNX operator '{node.op_type}' (node '{node.name}') is not "
            f"supported; supported: {sorted(OPS)}"
        )
    out = fn(node, inputs, opset, **kw)
    return list(out) if isinstance(out, (list, tuple)) else [out]


def run_float(graph, feeds, hook=None, keep_all=False):
    """Runs the whole graph in float32.

    Args:
        graph: :class:`~quantization.onnx_graph.OnnxGraph`.
        feeds: mapping ``input name -> array`` (a bare array feeds the single
            graph input).
        hook: optional ``hook(node, inputs, outputs)`` called per node.
        keep_all: return every intermediate tensor instead of graph outputs.
    """
    feeds = normalize_feeds(graph, feeds)
    env = dict(graph.initializers)
    env.update(feeds)
    for node in graph.nodes:
        args = [env[i] if i else None for i in node.inputs]
        outs = run_node(node, args, graph.opset)
        if hook is not None:
            hook(node, args, outs)
        env.update(zip(node.outputs, outs))
    if keep_all:
        return env
    result = [env[o] for o in graph.outputs]
    return result[0] if len(result) == 1 else result


def normalize_feeds(graph, feeds):
    """Maps a bare array / name-keyed dict to float32 feeds."""
    if not isinstance(feeds, dict):
        names = graph.input_names
        if len(names) != 1:
            raise ValueError(
                f"model has {len(names)} inputs; pass a dict keyed by {names}"
            )
        feeds = {names[0]: feeds}
    return {k: np.asarray(v, dtype=np.float32) for k, v in feeds.items()}


# ----------------------------------------------------------------- helpers --
def constant_value(node):
    """Value of an ONNX ``Constant`` node."""
    a = node.attrs
    for key in (
        "value",
        "value_float",
        "value_int",
        "value_floats",
        "value_ints",
    ):
        if key in a:
            v = np.asarray(a[key])
            if key == "value_float":
                v = v.astype(np.float32)
            elif key in ("value_int", "value_ints"):
                v = v.astype(np.int64)
            return v
    raise UnsupportedOp(f"Constant node '{node.name}' has no supported value")


def _norm_axis(axis, ndim):
    return axis + ndim if axis < 0 else axis


def _attr_or_input(node, inputs, idx, name, default=None):
    """Value from input ``idx`` (opset >= 13 style) or attribute ``name``."""
    if len(inputs) > idx and inputs[idx] is not None:
        return np.asarray(inputs[idx]).astype(np.int64).tolist()
    return node.attr(name, default)


def conv_pads(node, in_hw, kernel, strides, dilations):
    """Resolves ``pads`` / ``auto_pad`` into ``(top, left, bottom, right)``."""
    auto = node.attr("auto_pad", "NOTSET")
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        pads = []
        for size, k, s, d in zip(in_hw, kernel, strides, dilations):
            out = -(-size // s)
            total = max((out - 1) * s + d * (k - 1) + 1 - size, 0)
            small, big = total // 2, total - total // 2
            pads.append((small, big) if auto == "SAME_UPPER" else (big, small))
        return (pads[0][0], pads[1][0], pads[0][1], pads[1][1])
    if auto == "VALID":
        return (0, 0, 0, 0)
    p = node.attr("pads", [0] * 4)
    return (p[0], p[1], p[2], p[3])


# ------------------------------------------------------------- linear ops --
@register("Conv")
def _conv(node, inputs, opset, **kw):
    x, w = inputs[0], inputs[1]
    if x.ndim != 4:
        raise UnsupportedOp(
            f"Conv '{node.name}': only 2-D convolution is supported"
        )
    strides = tuple(node.attr("strides", [1, 1]))
    dilations = tuple(node.attr("dilations", [1, 1]))
    pads = conv_pads(node, x.shape[2:], w.shape[2:], strides, dilations)
    y = native.conv2d_f32(
        x, w, strides, pads, dilations, node.attr("group", 1)
    )
    if len(inputs) > 2 and inputs[2] is not None:
        y = y + inputs[2].reshape(1, -1, 1, 1)
    return y


@register("Gemm")
def _gemm(node, inputs, opset, **kw):
    a, b = inputs[0], inputs[1]
    if node.attr("transA", 0):
        a = a.T
    if node.attr("transB", 0):
        b = b.T
    y = np.float32(node.attr("alpha", 1.0)) * native.gemm_f32(a, b)
    if len(inputs) > 2 and inputs[2] is not None:
        y = y + np.float32(node.attr("beta", 1.0)) * inputs[2]
    return y.astype(np.float32)


@register("MatMul")
def _matmul(node, inputs, opset, **kw):
    return np.matmul(inputs[0], inputs[1]).astype(np.float32)


@register("LinearLayer")
def _linear_layer(node, inputs, opset, **kw):
    """``MatMul(x, W) + b``: one layer with the bias fused in."""
    return (np.matmul(inputs[0], inputs[1]) + inputs[2]).astype(np.float32)


@register("FusedElementwise")
def _fused(node, inputs, opset, **kw):
    """Evaluates the element-wise subgraph collected by ``graph_passes``."""
    env = dict(node.attrs["consts"])
    env[node.inputs[0]] = inputs[0]
    for sub in node.attrs["nodes"]:
        args = [env[i] if i else None for i in sub.inputs]
        env[sub.outputs[0]] = run_node(sub, args, opset)[0]
    return env[node.outputs[0]]


@register("BatchNormalization")
def _bn(node, inputs, opset, **kw):
    x, gamma, beta, mean, var = inputs[:5]
    shape = [1, -1] + [1] * (x.ndim - 2)
    a = gamma / np.sqrt(var + node.attr("epsilon", 1e-5))
    b = beta - a * mean
    return (x * a.reshape(shape) + b.reshape(shape)).astype(np.float32)


# ------------------------------------------------------------ element-wise --
def _binary(fn):
    def op(node, inputs, opset, **kw):
        return fn(inputs[0], inputs[1])

    return op


OPS["Add"] = _binary(np.add)
OPS["Sub"] = _binary(np.subtract)
OPS["Mul"] = _binary(np.multiply)
OPS["Div"] = _binary(np.divide)
OPS["Max"] = _binary(np.maximum)
OPS["Min"] = _binary(np.minimum)


@unary("Sigmoid")
def _sigmoid(node, x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


@unary("Relu")
def _relu(node, x):
    return np.maximum(x, 0)


@unary("LeakyRelu")
def _leaky(node, x):
    return np.where(x >= 0, x, node.attr("alpha", 0.01) * x).astype(np.float32)


@unary("Tanh")
def _tanh(node, x):
    return np.tanh(x).astype(np.float32)


@unary("HardSigmoid")
def _hard_sigmoid(node, x):
    a, b = node.attr("alpha", 0.2), node.attr("beta", 0.5)
    return np.clip(a * x + b, 0, 1).astype(np.float32)


@unary("HardSwish")
def _hard_swish(node, x):
    return (x * np.clip(x / 6.0 + 0.5, 0, 1)).astype(np.float32)


@unary("Silu")
def _silu(node, x):
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(np.float32)


@unary("Mish")
def _mish(node, x):
    x64 = x.astype(np.float64)
    return (x64 * np.tanh(np.logaddexp(0, x64))).astype(np.float32)


@unary("Elu")
def _elu(node, x):
    a = node.attr("alpha", 1.0)
    return np.where(x > 0, x, a * np.expm1(np.minimum(x, 0))).astype(
        np.float32
    )


@unary("Softplus")
def _softplus(node, x):
    return np.logaddexp(0, x.astype(np.float64)).astype(np.float32)


@unary("Abs")
def _abs(node, x):
    return np.abs(x)


@unary("Neg")
def _neg(node, x):
    return -x


@unary("Sqrt")
def _sqrt(node, x):
    return np.sqrt(x).astype(np.float32)


@unary("Exp")
def _exp(node, x):
    return np.exp(x.astype(np.float64)).astype(np.float32)


def _erf(x):
    """erf via Abramowitz-Stegun 7.1.26 refined by one Newton-free series
    split: |error| < 1.5e-7, below float32 resolution of the activations."""
    x = x.astype(np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (
        0.254829592
        + t
        * (
            -0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))
        )
    )
    return sign * (1.0 - poly * np.exp(-ax * ax))


@unary("Erf")
def _erf_op(node, x):
    return _erf(x).astype(np.float32)


@unary("Gelu")
def _gelu(node, x):
    x64 = x.astype(np.float64)
    if node.attr("approximate", "none") == "tanh":
        inner = np.sqrt(2 / np.pi) * (x64 + 0.044715 * x64**3)
        return (0.5 * x64 * (1 + np.tanh(inner))).astype(np.float32)
    return (0.5 * x64 * (1 + _erf(x64 / np.sqrt(2)))).astype(np.float32)


@register("Pow")
def _pow(node, inputs, opset, **kw):
    return np.power(
        inputs[0].astype(np.float64), np.asarray(inputs[1]).astype(np.float64)
    ).astype(np.float32)


@register("Expand")
def _expand(node, inputs, opset, **kw):
    shape = [int(v) for v in np.asarray(inputs[1]).reshape(-1)]
    return np.broadcast_to(
        inputs[0], np.broadcast_shapes(inputs[0].shape, tuple(shape))
    ).copy()


@register("LayerNormalization")
def _layer_norm(node, inputs, opset, **kw):
    x = inputs[0].astype(np.float64)
    axis = _norm_axis(node.attr("axis", -1), x.ndim)
    axes = tuple(range(axis, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
    y = (x - mean) / np.sqrt(var + node.attr("epsilon", 1e-5))
    y = y * inputs[1]
    if len(inputs) > 2 and inputs[2] is not None:
        y = y + inputs[2]
    return y.astype(np.float32)


@register("Clip")
def _clip(node, inputs, opset, **kw):
    lo = (
        inputs[1]
        if len(inputs) > 1 and inputs[1] is not None
        else node.attr("min")
    )
    hi = (
        inputs[2]
        if len(inputs) > 2 and inputs[2] is not None
        else node.attr("max")
    )
    return np.clip(inputs[0], lo, hi)


def clip_bounds(node, graph_consts):
    """``(min, max)`` of a Clip node resolved against folded constants."""
    lo = node.attr("min")
    hi = node.attr("max")
    if len(node.inputs) > 1 and node.inputs[1]:
        lo = float(graph_consts[node.inputs[1]])
    if len(node.inputs) > 2 and node.inputs[2]:
        hi = float(graph_consts[node.inputs[2]])
    return lo, hi


def _softmax_axis(node, opset, ndim):
    return _norm_axis(node.attr("axis", -1 if opset >= 13 else 1), ndim)


@register("Softmax")
def _softmax(node, inputs, opset, **kw):
    x = inputs[0].astype(np.float64)
    axis = _softmax_axis(node, opset, x.ndim)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


@register("LogSoftmax")
def _log_softmax(node, inputs, opset, **kw):
    x = inputs[0].astype(np.float64)
    axis = _softmax_axis(node, opset, x.ndim)
    z = x - x.max(axis=axis, keepdims=True)
    return (z - np.log(np.exp(z).sum(axis=axis, keepdims=True))).astype(
        np.float32
    )


@register("ConstantOfShape")
def _constant_of_shape(node, inputs, opset, **kw):
    value = node.attr("value")
    value = np.zeros(1, np.float32) if value is None else np.asarray(value)
    shape = [int(v) for v in np.asarray(inputs[0]).reshape(-1)]
    return np.full(shape, value.reshape(-1)[0], dtype=value.dtype)


@register("Equal")
def _equal(node, inputs, opset, **kw):
    return np.equal(inputs[0], inputs[1])


@register("Less")
def _less(node, inputs, opset, **kw):
    return np.less(inputs[0], inputs[1])


@register("Greater")
def _greater(node, inputs, opset, **kw):
    return np.greater(inputs[0], inputs[1])


@register("Not")
def _not(node, inputs, opset, **kw):
    return np.logical_not(inputs[0])


@register("Where")
def _where(node, inputs, opset, **kw):
    return np.where(inputs[0], inputs[1], inputs[2])


@register("Cast")
def _cast(node, inputs, opset, **kw):
    from onnx import helper

    return inputs[0].astype(helper.tensor_dtype_to_np_dtype(node.attr("to")))


# ------------------------------------------------------------------ shapes --
@register("Identity", "Dropout")
def _identity(node, inputs, opset, **kw):
    return inputs[0]


@register("Reshape")
def _reshape(node, inputs, opset, **kw):
    x = inputs[0]
    shape = [int(s) for s in np.asarray(inputs[1]).reshape(-1)]
    if not node.attr("allowzero", 0):
        shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
    return x.reshape(shape)


@register("Flatten")
def _flatten(node, inputs, opset, **kw):
    x = inputs[0]
    axis = _norm_axis(node.attr("axis", 1), x.ndim)
    return x.reshape(int(np.prod(x.shape[:axis])), -1)


@register("Squeeze")
def _squeeze(node, inputs, opset, **kw):
    axes = _attr_or_input(node, inputs, 1, "axes")
    x = inputs[0]
    if axes is None:
        return np.squeeze(x)
    return np.squeeze(x, axis=tuple(_norm_axis(a, x.ndim) for a in axes))


@register("Unsqueeze")
def _unsqueeze(node, inputs, opset, **kw):
    axes = _attr_or_input(node, inputs, 1, "axes")
    x = inputs[0]
    out_ndim = x.ndim + len(axes)
    for a in sorted(_norm_axis(a, out_ndim) for a in axes):
        x = np.expand_dims(x, a)
    return x


@register("Transpose")
def _transpose(node, inputs, opset, **kw):
    perm = node.attr("perm")
    return np.transpose(inputs[0], perm if perm else None)


@register("Concat")
def _concat(node, inputs, opset, **kw):
    return np.concatenate(inputs, axis=node.attr("axis"))


@register("Split")
def _split(node, inputs, opset, **kw):
    x = inputs[0]
    axis = _norm_axis(node.attr("axis", 0), x.ndim)
    sizes = _attr_or_input(node, inputs, 1, "split")
    n_out = len(node.outputs)
    if not sizes:
        sizes = [x.shape[axis] // n_out] * n_out
    idx = np.cumsum(sizes)[:-1]
    return list(np.split(x, idx, axis=axis))


@register("Slice")
def _slice(node, inputs, opset, **kw):
    x = inputs[0]
    starts = _attr_or_input(node, inputs, 1, "starts")
    ends = _attr_or_input(node, inputs, 2, "ends")
    axes = _attr_or_input(node, inputs, 3, "axes", list(range(len(starts))))
    steps = _attr_or_input(node, inputs, 4, "steps", [1] * len(starts))
    index = [slice(None)] * x.ndim
    for s, e, a, st in zip(starts, ends, axes, steps):
        index[_norm_axis(a, x.ndim)] = slice(
            None if s >= 2**62 else s, None if abs(e) >= 2**62 else e, st
        )
    return x[tuple(index)]


@register("Shape")
def _shape(node, inputs, opset, **kw):
    shape = np.array(inputs[0].shape, dtype=np.int64)
    return shape[node.attr("start", 0) : node.attr("end", None)]


@register("Gather")
def _gather(node, inputs, opset, **kw):
    return np.take(
        inputs[0],
        np.asarray(inputs[1]).astype(np.int64),
        axis=node.attr("axis", 0),
    )


@register("Pad")
def _pad(node, inputs, opset, pad_value=None, **kw):
    x = inputs[0]
    pads = _attr_or_input(node, inputs, 1, "pads")
    if pad_value is None:
        if len(inputs) > 2 and inputs[2] is not None:
            pad_value = np.asarray(inputs[2]).reshape(()).item()
        else:
            pad_value = node.attr("value", 0)
    n = x.ndim
    width = [(pads[i], pads[i + n]) for i in range(n)]
    mode = node.attr("mode", "constant")
    if mode == "constant":
        return np.pad(x, width, mode="constant", constant_values=pad_value)
    return np.pad(x, width, mode={"reflect": "reflect", "edge": "edge"}[mode])


# ----------------------------------------------------------------- pooling --
def _pool_geometry(node, x):
    kernel = tuple(node.attr("kernel_shape"))
    strides = tuple(node.attr("strides", [1] * len(kernel)))
    dilations = tuple(node.attr("dilations", [1] * len(kernel)))
    if x.ndim != 4:
        raise UnsupportedOp(
            f"{node.op_type} '{node.name}': only 2-D pooling is supported"
        )
    pads = conv_pads(node, x.shape[2:], kernel, strides, dilations)
    if node.attr("ceil_mode", 0):
        # Grow the end padding so the last partial window is included.
        pads = list(pads)
        for ax in range(2):
            size = x.shape[2 + ax] + pads[ax] + pads[ax + 2]
            ek = dilations[ax] * (kernel[ax] - 1) + 1
            out = -(-(size - ek) // strides[ax]) + 1
            pads[ax + 2] += max((out - 1) * strides[ax] + ek - size, 0)
    return kernel, strides, dilations, tuple(pads)


def _windows(xp, kernel, strides, dilations):
    ek = [d * (k - 1) + 1 for k, d in zip(kernel, dilations)]
    win = sliding_window_view(xp, tuple(ek), axis=(2, 3))
    return win[
        :, :, :: strides[0], :: strides[1], :: dilations[0], :: dilations[1]
    ]


@register("MaxPool")
def _maxpool(node, inputs, opset, pad_value=None, **kw):
    x = inputs[0]
    kernel, strides, dilations, pads = _pool_geometry(node, x)
    if pad_value is None:
        pad_value = (
            -np.inf
            if np.issubdtype(x.dtype, np.floating)
            else np.iinfo(x.dtype).min
        )
    xp = np.pad(
        x,
        ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
        constant_values=pad_value,
    )
    return _windows(xp, kernel, strides, dilations).max(axis=(4, 5))


@register("AveragePool")
def _avgpool(node, inputs, opset, **kw):
    x = inputs[0].astype(np.float32)
    kernel, strides, _, pads = _pool_geometry(node, x)
    dil = (1, 1)
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    total = _windows(xp, kernel, strides, dil).sum(axis=(4, 5))
    if node.attr("count_include_pad", 0):
        return (total / (kernel[0] * kernel[1])).astype(np.float32)
    ones = np.pad(
        np.ones_like(x[:1, :1]),
        ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
    )
    cnt = _windows(ones, kernel, strides, dil).sum(axis=(4, 5))
    return (total / cnt).astype(np.float32)


@register("GlobalAveragePool")
def _gap(node, inputs, opset, **kw):
    x = inputs[0]
    return x.mean(axis=tuple(range(2, x.ndim)), keepdims=True).astype(
        np.float32
    )


@register("GlobalMaxPool")
def _gmp(node, inputs, opset, **kw):
    x = inputs[0]
    return x.max(axis=tuple(range(2, x.ndim)), keepdims=True)


@register("ReduceMean")
def _reduce_mean(node, inputs, opset, **kw):
    x = inputs[0]
    axes = _attr_or_input(node, inputs, 1, "axes")
    axes = tuple(_norm_axis(a, x.ndim) for a in axes) if axes else None
    return x.mean(axis=axes, keepdims=bool(node.attr("keepdims", 1))).astype(
        np.float32
    )
