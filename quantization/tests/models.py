"""Builders for small ONNX test models (no deep-learning framework needed)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class ModelBuilder:
    """Tiny helper that accumulates nodes / initializers of an ONNX graph."""

    def __init__(self, seed=0, opset=13):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.inits = []
        self.opset = opset
        self._n = 0

    def _name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, value, base="const"):
        name = self._name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def weight(self, *shape, scale=0.3, base="w"):
        return self.const(
            (self.rng.standard_normal(shape) * scale).astype(np.float32), base
        )

    def op(self, op_type, inputs, n_out=1, **attrs):
        outs = [self._name(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(
            helper.make_node(
                op_type, inputs, outs, name=self._name(op_type), **attrs
            )
        )
        return outs[0] if n_out == 1 else outs

    def conv(
        self,
        x,
        cin,
        cout,
        k=3,
        stride=1,
        pad=None,
        groups=1,
        bias=True,
        dilation=1,
    ):
        pad = k // 2 if pad is None else pad
        inputs = [x, self.weight(cout, cin // groups, k, k)]
        if bias:
            inputs.append(self.weight(cout, scale=0.1, base="b"))
        return self.op(
            "Conv",
            inputs,
            kernel_shape=[k, k],
            strides=[stride, stride],
            pads=[pad] * 4,
            dilations=[dilation, dilation],
            group=groups,
        )

    def silu(self, x):
        return self.op("Mul", [x, self.op("Sigmoid", [x])])

    def build(self, input_name, input_shape, output, output_shape=None):
        graph = helper.make_graph(
            self.nodes,
            "test",
            [
                helper.make_tensor_value_info(
                    input_name, TensorProto.FLOAT, input_shape
                )
            ],
            [
                helper.make_tensor_value_info(
                    output, TensorProto.FLOAT, output_shape
                )
            ],
            self.inits,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset)]
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model


def build_yolo_like(seed=0, size=32, classes=10, width=8):
    """A miniature YOLOv8-cls: Conv-SiLU stem, C2f-style block, GAP, Gemm."""
    b = ModelBuilder(seed)
    x = "images"
    y = b.silu(b.conv(x, 3, width, 3, stride=2))
    y = b.silu(b.conv(y, width, 2 * width, 3, stride=2))
    # C2f: split -> bottleneck -> concat
    a, c = b.op(
        "Split",
        [y, b.const(np.array([width, width], dtype=np.int64))],
        n_out=2,
        axis=1,
    )
    d = b.silu(b.conv(c, width, width, 3))
    d = b.op("Add", [c, b.silu(b.conv(d, width, width, 3))])
    y = b.op("Concat", [a, c, d], axis=1)
    y = b.silu(b.conv(y, 3 * width, 4 * width, 1))
    # depthwise + pointwise
    y = b.silu(b.conv(y, 4 * width, 4 * width, 3, groups=4 * width))
    y = b.silu(b.conv(y, 4 * width, 8 * width, 1))
    y = b.op("GlobalAveragePool", [y])
    y = b.op("Flatten", [y], axis=1)
    y = b.op(
        "Gemm",
        [y, b.weight(classes, 8 * width), b.weight(classes, scale=0.1)],
        transB=1,
    )
    y = b.op("Softmax", [y], axis=1)
    return b.build(x, ["N", 3, size, size], y, ["N", classes])


def _layer_norm(b, x, dim, style):
    """LayerNorm over the last axis: fused op or PyTorch decomposition."""
    gamma = b.const((1 + 0.1 * b.rng.standard_normal(dim)).astype(np.float32))
    beta = b.const((0.1 * b.rng.standard_normal(dim)).astype(np.float32))
    if style == "op":
        return b.op(
            "LayerNormalization", [x, gamma, beta], axis=-1, epsilon=1e-6
        )
    mean = b.op("ReduceMean", [x], axes=[-1], keepdims=1)
    d = b.op("Sub", [x, mean])
    var = b.op(
        "ReduceMean",
        [b.op("Pow", [d, b.const(np.float32(2.0))])],
        axes=[-1],
        keepdims=1,
    )
    std = b.op("Sqrt", [b.op("Add", [var, b.const(np.float32(1e-6))])])
    return b.op("Add", [b.op("Mul", [b.op("Div", [d, std]), gamma]), beta])


def _gelu(b, x, style):
    if style == "op":
        return b.op("Gelu", [x])
    s2 = b.const(np.float32(np.sqrt(2.0)))
    erf = b.op("Erf", [b.op("Div", [x, s2])])
    one_plus = b.op("Add", [erf, b.const(np.float32(1.0))])
    return b.op("Mul", [b.op("Mul", [x, b.const(np.float32(0.5))]), one_plus])


def _linear(b, x, cin, cout, scale=0.15):
    """3-D Linear as exported by PyTorch: MatMul + Add."""
    y = b.op("MatMul", [x, b.weight(cin, cout, scale=scale)])
    return b.op("Add", [y, b.weight(cout, scale=0.05, base="bias")])


def build_vit(
    seed=0,
    img=32,
    patch=8,
    dim=32,
    heads=4,
    depth=2,
    classes=10,
    batch=1,
    ln_style="decomposed",
    gelu_style="decomposed",
    qk_scale="logits",
):
    """A small Vision Transformer with the structure of ``timm`` exports.

    Patch embedding (Conv), cls token + positional embedding, ``depth`` pre-LN
    blocks (multi-head self-attention with activation x activation MatMuls,
    GELU MLP), final LayerNorm and a linear head on the cls token. The cls
    token is expanded with a ``Shape -> Gather -> Unsqueeze -> Concat`` chain
    like PyTorch exports do (folded because the batch size is static).
    ``qk_scale`` is ``"logits"`` (scale the attention logits) or
    ``"separate"`` (scale ``q`` and ``k^T``, as timm exports do).
    """
    b = ModelBuilder(seed, opset=17 if "op" in (ln_style, gelu_style) else 13)
    if gelu_style == "op":
        b.opset = 20
    hd = dim // heads
    tokens = (img // patch) ** 2 + 1

    x = b.conv("images", 3, dim, patch, stride=patch, pad=0)
    x = b.op("Reshape", [x, b.const(np.array([batch, dim, -1], np.int64))])
    x = b.op("Transpose", [x], perm=[0, 2, 1])
    cls = b.weight(1, 1, dim, scale=0.1, base="cls")
    bsz = b.op(
        "Gather",
        [b.op("Shape", ["images"]), b.const(np.array(0, np.int64))],
        axis=0,
    )
    shape = b.op(
        "Concat",
        [
            b.op("Unsqueeze", [bsz, b.const(np.array([0], np.int64))]),
            b.const(np.array([1, dim], np.int64)),
        ],
        axis=0,
    )
    x = b.op("Concat", [b.op("Expand", [cls, shape]), x], axis=1)
    x = b.op("Add", [x, b.weight(1, tokens, dim, scale=0.1, base="pos")])

    for _ in range(depth):
        h = _layer_norm(b, x, dim, ln_style)
        qkv = _linear(b, h, dim, 3 * dim)
        qkv = b.op(
            "Reshape",
            [qkv, b.const(np.array([batch, tokens, 3, heads, hd], np.int64))],
        )
        qkv = b.op("Transpose", [qkv], perm=[2, 0, 3, 1, 4])
        split_attrs = {"num_outputs": 3} if b.opset >= 18 else {}
        q, k, v = [
            b.op("Squeeze", [t, b.const(np.array([0], np.int64))])
            for t in b.op("Split", [qkv], n_out=3, axis=0, **split_attrs)
        ]
        kt = b.op("Transpose", [k], perm=[0, 1, 3, 2])
        if qk_scale == "separate":  # timm: q and k^T each get hd**-0.25
            s4 = b.const(np.float32(hd**-0.25))
            q, kt = b.op("Mul", [q, s4]), b.op("Mul", [kt, s4])
            att = b.op("MatMul", [q, kt])
        else:  # original ViT: the scale is applied to the logits
            att = b.op("MatMul", [q, kt])
            att = b.op("Mul", [att, b.const(np.float32(hd**-0.5))])
        att = b.op("Softmax", [att], axis=-1)
        o = b.op("Transpose", [b.op("MatMul", [att, v])], perm=[0, 2, 1, 3])
        o = b.op(
            "Reshape", [o, b.const(np.array([batch, tokens, dim], np.int64))]
        )
        x = b.op("Add", [x, _linear(b, o, dim, dim)])

        h = _layer_norm(b, x, dim, ln_style)
        h = _gelu(b, _linear(b, h, dim, 4 * dim), gelu_style)
        x = b.op("Add", [x, _linear(b, h, 4 * dim, dim)])

    x = _layer_norm(b, x, dim, ln_style)
    x = b.op("Gather", [x, b.const(np.array(0, np.int64))], axis=1)
    y = b.op(
        "Gemm",
        [x, b.weight(classes, dim), b.weight(classes, scale=0.05)],
        transB=1,
    )
    return b.build("images", [batch, 3, img, img], y, [batch, classes])


def attention_core(b, qkv, batch, tokens, heads, hd):
    """The attention core as exported by PyTorch (no projections).

    ``qkv`` is a ``[batch, tokens, 3 * heads * hd]`` tensor; the result is
    ``[batch, tokens, heads * hd]``.
    """
    split_attrs = {"num_outputs": 3} if b.opset >= 18 else {}
    t = b.op(
        "Transpose",
        [
            b.op(
                "Reshape",
                [
                    qkv,
                    b.const(np.array([batch, tokens, 3, heads, hd], np.int64)),
                ],
            )
        ],
        perm=[2, 0, 3, 1, 4],
    )
    q, k, v = [
        b.op("Squeeze", [s, b.const(np.array([0], np.int64))])
        for s in b.op("Split", [t], n_out=3, axis=0, **split_attrs)
    ]
    att = b.op("MatMul", [q, b.op("Transpose", [k], perm=[0, 1, 3, 2])])
    o = b.op("MatMul", [b.op("Softmax", [att], axis=-1), v])
    o = b.op("Transpose", [o], perm=[0, 2, 1, 3])
    return b.op(
        "Reshape",
        [o, b.const(np.array([batch, tokens, heads * hd], np.int64))],
    )
