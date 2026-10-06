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


# ------------------------------------------------------- tidy-up passes ----
def _wrap(b, out, shape, opset=13, in_name="x"):
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)]
    )
    model.ir_version = 8
    return model


def _ops(graph):
    return [n.op_type for n in graph.nodes]


def test_noop_cast_is_removed_but_real_casts_stay():
    b = ModelBuilder(0)
    y = b.op(
        "Cast",
        [b.op("Cast", ["x"], to=TensorProto.FLOAT)],
        to=TensorProto.FLOAT,
    )
    y = b.op("Relu", [y])
    g = OnnxGraph.from_model(_wrap(b, y, [2, 3]))
    assert _ops(g) == ["Relu"] and g.fusions["cast"] == 2

    b = ModelBuilder(0)
    y = b.op("Relu", [b.op("Cast", ["x"], to=TensorProto.DOUBLE)])
    assert "Cast" in _ops(OnnxGraph.from_model(_wrap(b, y, [2, 3])))


def test_timm_graph_casts_are_folded_with_the_shape_chain():
    # Shape -> Slice -> Cast -> Sqrt -> Div computing the attention scale
    b = ModelBuilder(0)
    shp = b.op("Shape", ["x"])
    last = b.op(
        "Slice",
        [
            shp,
            b.const(np.array([-1], np.int64)),
            b.const(np.array([2**62], np.int64)),
        ],
    )
    scale = b.op(
        "Div",
        [
            b.const(np.float32(1.0)),
            b.op("Sqrt", [b.op("Cast", [last], to=TensorProto.FLOAT)]),
        ],
    )
    y = b.op("Mul", ["x", scale])
    g = OnnxGraph.from_model(_wrap(b, y, [2, 9, 16]))
    assert not {"Shape", "Slice", "Cast", "Sqrt", "Div"} & set(_ops(g))
    x = np.random.default_rng(0).standard_normal((2, 9, 16)).astype(np.float32)
    np.testing.assert_allclose(
        _float(g, x), _ort(_wrap(b, y, [2, 9, 16]), x, "x"), atol=1e-6
    )


@pytest.mark.parametrize("style", ["decomposed", "op"])
def test_gelu_is_recognised_by_name(style):
    model = build_vit(depth=1, ln_style="op", gelu_style=style)
    g = OnnxGraph.from_model(model)
    assert _ops(g).count("Gelu") == 1
    assert "FusedElementwise" not in _ops(g)


def test_tanh_gelu_and_silu_and_unknown_functions():
    x = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)

    def silu_model():
        b = ModelBuilder(0)
        y = b.silu("x")
        return _wrap(b, y, [2, 8])

    g = OnnxGraph.from_model(silu_model())
    assert _ops(g) == ["Silu"] and g.fusions["activation"] == 1
    np.testing.assert_allclose(
        _float(g, x), _ort(silu_model(), x, "x"), atol=1e-6
    )

    b = ModelBuilder(0)  # sigmoid(x) * 2 + 1 is not a known activation
    y = b.op(
        "Add",
        [
            b.op("Mul", [b.op("Sigmoid", ["x"]), b.const(np.float32(2.0))]),
            b.const(np.float32(1.0)),
        ],
    )
    g = OnnxGraph.from_model(_wrap(b, y, [2, 8]))
    assert _ops(g) == ["FusedElementwise"]


def test_chains_of_constant_ops_are_merged():
    b = ModelBuilder(0)
    y = b.op(
        "Mul",
        [
            b.op(
                "Mul",
                [
                    b.op(
                        "Add",
                        [
                            b.op("Add", ["x", b.const(np.float32(0.5))]),
                            b.const(np.float32(1.5)),
                        ],
                    ),
                    b.const(np.float32(2.0)),
                ],
            ),
            b.const(np.float32(-3.0)),
        ],
    )
    y = b.op("Mul", [y, b.const(np.float32(1.0))])  # neutral
    model = _wrap(b, y, [2, 8])
    x = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
    g = OnnxGraph.from_model(model, optimize=False)
    from quantization import graph_passes as gp

    removed = gp.merge_constant_ops(g)
    assert removed >= 3 and _ops(g).count("Mul") == 1
    assert _ops(g).count("Add") == 1
    np.testing.assert_allclose(_float(g, x), _ort(model, x, "x"), atol=1e-6)


@pytest.mark.parametrize(
    "kind",
    ["conv_scalar", "conv_channel", "matmul", "matmul_bias", "gemm_transb"],
)
def test_constant_multiplication_is_folded_into_the_linear_layer(kind):
    rng = np.random.default_rng(0)
    b = ModelBuilder(1)
    if kind.startswith("conv"):
        y = b.conv("x", 3, 5, 3)
        c = (
            np.float32(1.7)
            if kind == "conv_scalar"
            else rng.standard_normal((5, 1, 1)).astype(np.float32)
        )
        shape = [2, 3, 8, 8]
    elif kind.startswith("matmul"):
        y = b.op("MatMul", ["x", b.weight(6, 4)])
        if kind == "matmul_bias":
            y = b.op("Add", [y, b.weight(4, scale=0.5)])
        c = rng.standard_normal(4).astype(np.float32)
        shape = [3, 6]
    else:
        y = b.op(
            "Gemm", ["x", b.weight(4, 6), b.weight(4, scale=0.5)], transB=1
        )
        c = rng.standard_normal(4).astype(np.float32)
        shape = [3, 6]
    y = b.op("Mul", [y, b.const(c)])
    model = _wrap(b, y, shape)
    x = rng.standard_normal(shape).astype(np.float32)
    g = OnnxGraph.from_model(model)
    assert "Mul" not in _ops(g) and g.fusions["mul_into_linear"] == 1
    np.testing.assert_allclose(_float(g, x), _ort(model, x, "x"), atol=1e-5)


def test_mul_into_linear_skips_non_foldable_cases():
    b = ModelBuilder(1)
    y = b.conv("x", 3, 5, 3)
    y2 = b.op("Mul", [y, b.const(np.ones((1, 5, 8, 8), np.float32) * 2)])
    g = OnnxGraph.from_model(_wrap(b, y2, [2, 3, 8, 8]))
    assert g.fusions["mul_into_linear"] == 0 and "Mul" in _ops(g)


def test_timm_style_separate_q_and_k_scales_are_folded():
    model = build_vit(
        batch=2,
        depth=2,
        dim=32,
        heads=4,
        ln_style="op",
        gelu_style="op",
        qk_scale="separate",
    )
    g = OnnxGraph.from_model(model)
    assert g.fusions["attention_scale"] == 4  # q and k^T, 2 blocks
    assert "Mul" not in _ops(g)
    np.testing.assert_allclose(_float(g, X), _ort(model, X), atol=1e-5)


def test_logit_and_separate_scaling_give_the_same_function():
    a = build_vit(batch=2, depth=1, ln_style="op", gelu_style="op")
    b_ = build_vit(
        batch=2, depth=1, ln_style="op", gelu_style="op", qk_scale="separate"
    )
    ga, gb = OnnxGraph.from_model(a), OnnxGraph.from_model(b_)
    assert "Mul" not in _ops(ga) and "Mul" not in _ops(gb)
    np.testing.assert_allclose(_float(ga, X), _float(gb, X), atol=1e-5)
