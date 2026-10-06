"""ONNX loading and the NumPy reference operators (checked against ORT)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from quantization import OnnxGraph, float_ops
from quantization.tests.models import ModelBuilder, build_yolo_like


def _ort(model, x):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {sess.get_inputs()[0].name: x})


def _single(op, shape, extra=None, n_out=1, opset=13, **attrs):
    """Model with ``op`` applied to one input (+ constants in ``extra``)."""
    b = ModelBuilder(0, opset)
    inputs = ["x"] + [b.const(v) for v in (extra or [])]
    out = b.op(op, inputs, n_out=n_out, **attrs)
    outs = [out] if n_out == 1 else out
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
            for o in outs
        ],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)]
    )
    model.ir_version = 8
    return model


def _check(model, x, atol=1e-5):
    graph = OnnxGraph.from_model(model)
    got = float_ops.run_float(graph, x)
    got = got if isinstance(got, list) else [got]
    for g, r in zip(got, _ort(model, x)):
        np.testing.assert_allclose(g, r, atol=atol, rtol=1e-5)


X4 = np.random.default_rng(0).standard_normal((2, 3, 9, 8)).astype(np.float32)


def test_whole_model_matches_onnxruntime():
    model = build_yolo_like()
    x = (
        np.random.default_rng(1)
        .uniform(0, 1, (3, 3, 32, 32))
        .astype(np.float32)
    )
    _check(model, x, atol=1e-6)


@pytest.mark.parametrize(
    "op", ["Sigmoid", "Relu", "Tanh", "HardSwish", "Abs", "Neg", "Exp"]
)
def test_unary_ops(op):
    _check(_single(op, X4.shape, opset=14), X4)


def test_leaky_relu_elu_hardsigmoid_clip():
    _check(_single("LeakyRelu", X4.shape, alpha=0.1), X4)
    _check(_single("Elu", X4.shape, alpha=1.5), X4)
    _check(_single("HardSigmoid", X4.shape, alpha=0.3, beta=0.4), X4)
    _check(_single("Clip", X4.shape, [np.float32(-0.5), np.float32(0.7)]), X4)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
def test_softmax(op):
    _check(_single(op, X4.shape, axis=1), X4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kernel_shape=[2, 2], strides=[2, 2]),
        dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 0, 1, 0]),
        dict(kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1),
    ],
)
@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
def test_pooling(op, kwargs):
    _check(_single(op, X4.shape, **kwargs), X4)


def test_average_pool_count_include_pad():
    _check(
        _single(
            "AveragePool",
            X4.shape,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            count_include_pad=1,
        ),
        X4,
    )


@pytest.mark.parametrize("op", ["GlobalAveragePool", "GlobalMaxPool"])
def test_global_pooling(op):
    _check(_single(op, X4.shape), X4)


def test_reduce_mean():
    _check(_single("ReduceMean", X4.shape, axes=[2, 3], keepdims=0), X4)


def test_shape_ops():
    _check(_single("Reshape", X4.shape, [np.array([0, -1], np.int64)]), X4)
    _check(_single("Flatten", X4.shape, axis=2), X4)
    _check(_single("Transpose", X4.shape, perm=[0, 2, 3, 1]), X4)
    _check(_single("Unsqueeze", X4.shape, [np.array([0], np.int64)]), X4)
    _check(
        _single(
            "Slice",
            X4.shape,
            [
                np.array([1, 2], np.int64),
                np.array([3, 7], np.int64),
                np.array([1, 3], np.int64),
                np.array([1, 2], np.int64),
            ],
        ),
        X4,
    )


def test_split_variants():
    _check(
        _single(
            "Split", X4.shape, [np.array([1, 2], np.int64)], n_out=2, axis=1
        ),
        X4,
    )
    _check(
        _single("Split", (2, 4, 3), n_out=2, axis=1),
        np.ones((2, 4, 3), np.float32),
    )


def test_pad_constant():
    pads = np.array([0, 0, 1, 2, 0, 0, 3, 0], np.int64)
    _check(_single("Pad", X4.shape, [pads, np.float32(0.5)]), X4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strides=[2, 2], pads=[1, 1, 1, 1]),
        dict(dilations=[2, 2], pads=[2, 2, 2, 2]),
        dict(auto_pad="SAME_UPPER", strides=[2, 2]),
        dict(auto_pad="SAME_LOWER", strides=[2, 2]),
        dict(pads=[0, 1, 2, 3]),
    ],
)
def test_conv_variants(kwargs):
    b = ModelBuilder(0)
    y = b.op(
        "Conv",
        ["x", b.weight(4, 3, 3, 3), b.weight(4)],
        kernel_shape=[3, 3],
        **kwargs,
    )
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, X4.shape)],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    _check(model, X4, atol=1e-4)


def test_batchnorm_gemm_matmul_add_mul():
    b = ModelBuilder(0)
    c = 3
    bn = b.op(
        "BatchNormalization",
        [
            "x",
            b.const(np.array([1.0, -2.0, 0.5], np.float32)),
            b.weight(c),
            b.weight(c),
            b.const(np.array([0.5, 1.0, 2.0], np.float32)),
        ],
        epsilon=1e-3,
    )
    y = b.op(
        "Mul",
        [
            b.op("Add", [bn, b.const(np.float32(0.3))]),
            b.const(np.float32(2.0)),
        ],
    )
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, X4.shape)],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    _check(model, X4)

    a = np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32)
    w = np.random.default_rng(3).standard_normal((6, 5)).astype(np.float32)
    _check(_single("MatMul", a.shape, [w]), a)
    _check(
        _single(
            "Gemm", a.shape, [w, np.ones(5, np.float32)], alpha=0.5, beta=2.0
        ),
        a,
    )


def test_opset_11_style_attributes():
    # Older opsets pass split / axes / pads as attributes.
    _check(
        _single("Split", X4.shape, n_out=2, axis=1, split=[1, 2], opset=11), X4
    )
    _check(
        _single("Squeeze", (2, 1, 3), axes=[1], opset=11),
        np.ones((2, 1, 3), np.float32),
    )
    _check(
        _single(
            "Pad",
            X4.shape,
            mode="constant",
            value=1.0,
            pads=[0, 0, 1, 1, 0, 0, 1, 1],
            opset=2,
        ),
        X4,
    )


def test_constant_nodes_are_folded():
    b = ModelBuilder(0)
    c = b.op(
        "Constant",
        [],
        value=helper.make_tensor("v", TensorProto.INT64, [2], [0, -1]),
    )
    y = b.op("Reshape", ["x", c])
    shape = b.op("Shape", [b.const(np.zeros((2, 3, 4), np.float32))])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, X4.shape)],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    g = OnnxGraph.from_model(model)
    assert [n.op_type for n in g.nodes] == ["Reshape"]
    assert g.is_constant(c) and g.is_constant(shape)
    np.testing.assert_array_equal(g.initializers[shape], [2, 3, 4])
    _check(model, X4)


def test_identity_and_dropout_are_removed():
    b = ModelBuilder(0)
    y = b.op("Relu", [b.op("Dropout", [b.op("Identity", ["x"])])])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, X4.shape)],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    g = OnnxGraph.from_model(model)
    assert [n.op_type for n in g.nodes] == ["Relu"]
    _check(model, X4)


def test_graph_is_toposorted_and_names_unique():
    model = build_yolo_like()
    # reverse the node order and blank every name
    nodes = list(reversed(model.graph.node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for n in model.graph.node:
        n.name = ""
    g = OnnxGraph.from_model(model)
    seen = set(g.initializers) | set(g.input_names)
    for node in g.nodes:
        assert all(i in seen for i in node.inputs if i)
        seen.update(node.outputs)
    assert len({n.name for n in g.nodes}) == len(g.nodes)


def test_loads_from_path(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(build_yolo_like(), str(path))
    g = OnnxGraph.from_model(path)
    assert g.input_names == ["images"]


def test_unsupported_operator_has_clear_error():
    model = _single("Det", X4.shape)
    g = OnnxGraph.from_model(model)
    with pytest.raises(float_ops.UnsupportedOp, match="Det"):
        float_ops.run_float(g, X4)


def test_new_ops_match_onnxruntime():
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    _check(_single("Erf", x.shape), x, atol=2e-7)
    _check(_single("Gelu", x.shape, opset=20), x, atol=2e-6)
    _check(
        _single("Gelu", x.shape, opset=20, approximate="tanh"), x, atol=2e-6
    )
    _check(_single("Pow", x.shape, [np.float32(2.0)]), x)
    _check(
        _single("Expand", (1, 1, 8), [np.array([2, 3, 8], np.int64)]),
        x[:1, :1],
    )
    b = ModelBuilder(0)
    y = b.op(
        "LayerNormalization",
        ["x", b.weight(8, scale=1.0), b.weight(8, scale=0.1)],
        axis=-1,
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    _check(model, x, atol=1e-5)


def test_timm_style_cls_token_shape_chain_is_folded():
    """cls-token shape chain (ConstantOfShape, Equal, Where), torch.onnx."""
    b = ModelBuilder(0)
    zeros = b.op(
        "ConstantOfShape",
        [b.const(np.array([3], np.int64))],
        value=helper.make_tensor("v", TensorProto.INT64, [1], [0]),
    )
    shape = b.const(np.array([1, -1, 8], np.int64))
    target = b.op(
        "Where", [b.op("Equal", [shape, b.const(np.int64(-1))]), zeros, shape]
    )
    cls = b.weight(1, 1, 8, base="cls")
    out = b.op("Concat", [b.op("Expand", [cls, target]), "x"], axis=1)
    x = np.random.default_rng(0).standard_normal((1, 4, 8)).astype(np.float32)
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    g = OnnxGraph.from_model(model)
    assert not {"ConstantOfShape", "Equal", "Where", "Expand"} & {
        n.op_type for n in g.nodes
    }
    _check(model, x)


def test_comparison_ops():
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    for op in ("Less", "Greater"):
        node = helper.make_node(op, ["x", "y"], ["o"])
        from quantization.onnx_graph import Node

        n = Node("n", op, ["x", "y"], ["o"], {})
        got = float_ops.run_node(n, [x, np.zeros_like(x)])[0]
        np.testing.assert_array_equal(
            got, (x < 0) if op == "Less" else (x > 0)
        )
        assert node is not None
