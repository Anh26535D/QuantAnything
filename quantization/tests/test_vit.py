"""A small Vision Transformer through the whole pipeline."""

from collections import Counter

import numpy as np
import onnxruntime as ort
import pytest

from quantization import OnnxGraph, QuantContainer, native
from quantization.tests.models import build_vit

BATCH = 4


@pytest.fixture(scope="module")
def model():
    return build_vit(batch=BATCH, depth=2, dim=32, heads=4, classes=10)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((BATCH, 3, 32, 32)).astype(np.float32)
        for _ in range(4)
    ]


def _ort(model, x):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {"images": x})[0]


def _cosine(a, b):
    return float(
        np.mean(
            np.sum(a * b, 1)
            / np.linalg.norm(a, axis=1)
            / np.linalg.norm(b, axis=1)
        )
    )


@pytest.mark.parametrize(
    "ln,gelu", [("decomposed", "decomposed"), ("op", "op")]
)
def test_float_graph_matches_onnxruntime(ln, gelu, data):
    model = build_vit(batch=BATCH, ln_style=ln, gelu_style=gelu)
    c = QuantContainer(model)
    np.testing.assert_allclose(
        c.run_graph(data[0], "float"), _ort(model, data[0]), atol=1e-5
    )


def test_pytorch_style_decomposition_is_fused():
    dec = OnnxGraph.from_model(build_vit(depth=2))
    fused = OnnxGraph.from_model(
        build_vit(depth=2, ln_style="op", gelu_style="op")
    )
    # 2 blocks -> 5 LayerNorms, 2 GELUs, 8 MatMul+bias pairs (qkv, proj, mlp)
    assert dec.fusions == {"matmul_bias": 8, "layernorm": 5, "elementwise": 2}
    ops = Counter(n.op_type for n in dec.nodes)
    assert ops["LayerNormalization"] == 5
    assert ops["FusedElementwise"] == 2
    assert not {"ReduceMean", "Pow", "Sqrt", "Div", "Erf"} & set(ops)
    # the decomposed export collapses to the same graph as the fused ops
    assert len(dec.nodes) == len(fused.nodes)


def test_static_shape_chain_is_folded(model):
    ops = {n.op_type for n in OnnxGraph.from_model(model).nodes}
    assert not {"Shape", "Expand", "Unsqueeze"} & ops


def test_fusion_can_be_disabled(model):
    plain = OnnxGraph.from_model(model, optimize=False)
    assert plain.fusions == {}
    assert len(plain.nodes) > len(OnnxGraph.from_model(model).nodes)
    x = np.random.default_rng(1).standard_normal((BATCH, 3, 32, 32))
    np.testing.assert_allclose(
        QuantContainer(plain).run_graph(x, "float"),
        QuantContainer(model).run_graph(x, "float"),
        atol=1e-5,
    )


def test_linear_layers_use_integer_gemm_with_fused_bias(model):
    c = QuantContainer(model)
    kinds = Counter(type(l).__name__ for l in c.layers)
    assert kinds["GemmQuantize"] == 9  # 8 MatMul+bias + head
    assert kinds["ActivationQuantize"] == 2  # GELU as a single LUT
    gemms = [l for l in c.layers if type(l).__name__ == "GemmQuantize"]
    assert all(
        l.node.op_type == "Gemm" or len(l.node.inputs) == 3 for l in gemms
    )


def test_int16_is_nearly_lossless(model, data):
    c = QuantContainer(model, "int16")
    c.calibrate(data[:3])
    ref = _ort(model, data[3])
    for mode in ("quantize", "integer"):
        out = c.run_graph(data[3], mode)
        assert _cosine(out, ref) > 0.999
        np.testing.assert_array_equal(out.argmax(1), ref.argmax(1))


def test_int8_runs_and_stays_correlated(model, data):
    # Naive per-tensor min/max int8 is known to be rough on transformers
    # (residual-stream outliers); this only guards against gross breakage.
    c = QuantContainer(model, "int8")
    c.calibrate(data[:3])
    ref = _ort(model, data[3])
    for mode in ("quantize", "integer"):
        assert _cosine(c.run_graph(data[3], mode), ref) > 0.8


def test_activation_by_activation_matmul_uses_generic_layer(model):
    c = QuantContainer(model)
    generic = [l for l in c.layers if type(l).__name__ == "BaseQuantLayer"]
    assert {"MatMul", "Softmax", "LayerNormalization"} <= {
        l.node.op_type for l in generic
    }


def test_backends_bit_exact_on_vit(model, data, monkeypatch):
    if not native.available():
        pytest.skip("native kernels missing")
    c = QuantContainer(model, "int8")
    c.calibrate(data[:2])
    fast = c.run_graph(data[3], "integer")
    monkeypatch.setattr(native, "_lib", None)
    np.testing.assert_array_equal(fast, c.run_graph(data[3], "integer"))


def test_save_load_roundtrip(model, data, tmp_path):
    c = QuantContainer(model, "int16")
    c.calibrate(data[:2])
    c.save(tmp_path / "vit.json")
    loaded = QuantContainer.load(model, tmp_path / "vit.json")
    np.testing.assert_array_equal(
        c.run_graph(data[3], "integer"), loaded.run_graph(data[3], "integer")
    )
