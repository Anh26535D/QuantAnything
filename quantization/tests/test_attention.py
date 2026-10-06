"""MultiHeadAttention: fusion, float op, quantized layer and export."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from quantization import (
    MultiHeadAttentionQuantize,
    OnnxGraph,
    QuantContainer,
    float_ops,
)
from quantization.tests.models import (
    ModelBuilder,
    attention_core,
    build_vit,
)

B, T, H, HD = 2, 9, 3, 8


def _core_model(opset=13, extra_user=False):
    b = ModelBuilder(0, opset=opset)
    y = attention_core(b, "x", B, T, H, HD)
    out, shape = y, [B, T, H * HD]
    if extra_user:  # a second consumer of the probabilities breaks the match
        probs = next(n for n in b.nodes if n.op_type == "Softmax").output[0]
        out, shape = b.op("Concat", [probs, probs], axis=-1), [B, H, T, 2 * T]
    return b.build("x", [B, T, 3 * H * HD], out, shape)


def _data(shape=(B, T, 3 * H * HD), seed=0):
    return (
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    )


def _ort(model, x, name="x"):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {name: x})[0]


def test_twelve_nodes_become_one_layer_with_the_same_function():
    model = _core_model()
    g = OnnxGraph.from_model(model, optimize=False)
    assert len(g.nodes) == 12
    opt = OnnxGraph.from_model(model)
    assert [n.op_type for n in opt.nodes] == ["MultiHeadAttention"]
    assert opt.fusions["attention"] == 1
    node = opt.nodes[0]
    assert (node.attr("num_heads"), node.attr("head_dim")) == (H, HD)
    x = _data()
    np.testing.assert_allclose(
        float_ops.run_float(opt, x), _ort(model, x), atol=1e-6
    )


def test_pattern_is_left_alone_when_probabilities_have_other_users():
    opt = OnnxGraph.from_model(_core_model(extra_user=True))
    assert opt.fusions["attention"] == 0
    assert "Softmax" in [n.op_type for n in opt.nodes]


def test_vit_attention_is_fused_for_both_scale_styles():
    for style in ("logits", "separate"):
        g = OnnxGraph.from_model(
            build_vit(batch=2, depth=3, qk_scale=style, ln_style="op")
        )
        assert g.fusions["attention"] == 3
        assert "Softmax" not in [n.op_type for n in g.nodes]


@pytest.mark.parametrize("quant_type", ["int8", "uint8", "int16"])
def test_integer_attention_matches_fake_quant(quant_type):
    model = _core_model()
    cal = [_data(seed=s) for s in (1, 2)]
    c = QuantContainer(model, quant_type)
    c.calibrate(cal)
    assert isinstance(c.layers[0], MultiHeadAttentionQuantize)
    x = cal[0]
    fake, integer = c.run_graph(x, "quantize"), c.run_graph(x, "integer")
    step = c.qparams[c.graph.outputs[0]].scale
    noise = 4 * np.finfo(np.float32).eps * np.abs(fake).max()
    assert np.abs(fake - integer).max() <= 2 * step + noise
    ref = c.run_graph(x, "float")
    tol = {"int16": 2e-3}.get(quant_type, 0.15) * np.abs(ref).max()
    assert np.abs(integer - ref).max() < tol


def test_probability_grid_is_fixed_and_covers_zero_to_one():
    c = QuantContainer(_core_model(), "int8")
    c.calibrate([_data()])
    layer = c.layers[0]
    assert layer.p_scale == pytest.approx(1 / 255)
    assert layer.p_zp == -128
    u = QuantContainer(_core_model(), "uint8")
    u.calibrate([_data()])
    assert u.layers[0].p_zp == 0


def test_backends_agree_bit_exactly():
    from quantization import native

    if not native.available():
        pytest.skip("native kernels missing")
    c = QuantContainer(_core_model(), "int16")
    c.calibrate([_data()])
    fast = c.run_graph(_data(), "integer")
    lib, native._lib = native._lib, None
    try:
        slow = c.run_graph(_data(), "integer")
    finally:
        native._lib = lib
    np.testing.assert_array_equal(fast, slow)


# -------------------------------------------------------------- export ----
@pytest.mark.parametrize("opset", [13, 17, 18])
@pytest.mark.parametrize("use_functions", [True, False])
def test_export_runs_in_onnxruntime(opset, use_functions):
    model = _core_model(opset=opset)
    exported = OnnxGraph.from_model(model).to_model(
        use_functions=use_functions
    )
    onnx.checker.check_model(exported)
    ops = [n.op_type for n in exported.graph.node]
    if use_functions:
        assert ops == ["MultiHeadAttention"]
        assert [f.name for f in exported.functions] == ["MultiHeadAttention"]
        call = exported.graph.node[0]
        assert call.domain == "quantanything"
    else:
        assert "MultiHeadAttention" not in ops and not exported.functions
        assert "Softmax" in ops
    x = _data()
    np.testing.assert_allclose(_ort(exported, x), _ort(model, x), atol=1e-5)


def test_exported_file_loads_back_and_is_stable(tmp_path):
    model = build_vit(batch=2, depth=2, ln_style="op", gelu_style="op")
    path = tmp_path / "vit_opt.onnx"
    g = OnnxGraph.from_model(model)
    g.save(path)
    again = OnnxGraph.from_model(str(path))
    assert [n.op_type for n in again.nodes] == [n.op_type for n in g.nodes]
    x = (
        np.random.default_rng(0)
        .standard_normal((2, 3, 32, 32))
        .astype(np.float32)
    )
    np.testing.assert_allclose(
        float_ops.run_float(again, x), _ort(model, x, "images"), atol=1e-5
    )
    assert "MultiHeadAttention" in [n.op_type for n in again.nodes]
