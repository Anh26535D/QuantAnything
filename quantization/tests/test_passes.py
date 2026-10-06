"""Exact graph simplifications must not change the float function."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from quantization import OnnxGraph, QuantContainer, float_ops
from quantization.tests.models import ModelBuilder, build_vit


def _float(graph, x):
    return float_ops.run_float(graph, x)


def _ort(model, x, name="images"):
    return ort.InferenceSession(model.SerializeToString()).run(
        None, {name: x}
    )[0]


@pytest.fixture(scope="module")
def vit():
    return build_vit(
        batch=2, depth=2, dim=32, heads=4, ln_style="op", gelu_style="op"
    )


X = np.random.default_rng(0).standard_normal((2, 3, 32, 32)).astype(np.float32)


def test_vit_simplifications_apply_and_preserve_the_function(vit):
    plain = OnnxGraph.from_model(vit, optimize=False)
    opt = OnnxGraph.from_model(vit)
    # depth 2: 2 attention scales, 1 gather hoist, 5 LayerNorm gains folded
    assert opt.fusions["attention_scale"] == 2
    assert opt.fusions["hoist_gather"] == 1
    assert opt.fusions["layernorm_affine"] == 5
    np.testing.assert_allclose(_float(opt, X), _ort(vit, X), atol=1e-5)
    np.testing.assert_allclose(_float(opt, X), _float(plain, X), atol=1e-5)
    assert len(opt.nodes) < len(plain.nodes)


def test_layernorm_gain_is_moved_into_the_linear_layers(vit):
    opt = OnnxGraph.from_model(vit)
    for node in opt.nodes:
        if node.op_type == "LayerNormalization":
            assert np.all(opt.initializers[node.inputs[1]] == 1)
            assert np.all(opt.initializers[node.inputs[2]] == 0)
    assert "Mul" not in {n.op_type for n in opt.nodes}


def test_hoisted_gather_normalizes_only_the_selected_token(vit):
    opt = OnnxGraph.from_model(vit)
    env = float_ops.run_float(opt, X, keep_all=True)
    final = [n for n in opt.nodes if n.op_type == "LayerNormalization"][-1]
    assert env[final.inputs[0]].shape == (2, 32)  # cls token only
    assert opt.producers[final.inputs[0]].op_type == "Gather"


def test_passes_do_not_fire_when_the_pattern_is_broken():
    # a LayerNorm whose output is also a graph output must not be rewritten
    b = ModelBuilder(0, opset=17)
    ln = b.op(
        "LayerNormalization",
        ["x", b.weight(8, scale=1.0), b.weight(8, scale=0.1)],
        axis=-1,
    )
    y = b.op("MatMul", [ln, b.weight(8, 4)])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 8])],
        [
            helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
            for o in (ln, y)
        ],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    assert OnnxGraph.from_model(model).fusions["layernorm_affine"] == 0


def test_conv_bn_folding():
    b = ModelBuilder(1)
    c = b.conv("x", 3, 6, 3, bias=True)
    y = b.op(
        "BatchNormalization",
        [
            c,
            b.const(np.array([1.5, -0.8, 0.3, -2.0, 1.0, 0.7], np.float32)),
            b.weight(6),
            b.weight(6),
            b.const(np.array([0.5, 1.0, 2.0, 0.7, 1.2, 0.9], np.float32)),
        ],
        epsilon=1e-3,
    )
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 8, 8])],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    x = (
        np.random.default_rng(0)
        .standard_normal((2, 3, 8, 8))
        .astype(np.float32)
    )
    opt = OnnxGraph.from_model(model)
    assert [n.op_type for n in opt.nodes] == ["Conv"]
    assert opt.fusions["conv_bn"] == 1
    np.testing.assert_allclose(_float(opt, x), _ort(model, x, "x"), atol=1e-4)
    # and it quantizes as a single integer conv
    c = QuantContainer(model, "int8")
    c.calibrate(x)
    assert type(c.layers[0]).__name__ == "ConvQuantize"
    ref = _ort(model, x, "x")
    assert (
        np.abs(c.run_graph(x, "integer") - ref).max()
        < 0.03 * np.abs(ref).max()
    )


def test_optimization_can_be_disabled(vit):
    assert OnnxGraph.from_model(vit, optimize=False).fusions == {}
