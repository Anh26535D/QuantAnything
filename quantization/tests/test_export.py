"""The optimized graph can be written back as a standard ONNX file."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

import optimize_onnx
from quantization import OnnxGraph, QuantContainer
from quantization.tests.models import ModelBuilder, build_vit, build_yolo_like

INTERNAL = {"FusedElementwise", "Silu"}


def _run(model, x, name="images"):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {name: x})[0]


def test_vit_roundtrip_is_standard_and_equivalent():
    model = build_vit(batch=2, depth=2, qk_scale="separate")
    x = (
        np.random.default_rng(0)
        .standard_normal((2, 3, 32, 32))
        .astype(np.float32)
    )
    exported = OnnxGraph.from_model(model).to_model()
    onnx.checker.check_model(exported)
    ops = [n.op_type for n in exported.graph.node]
    assert not INTERNAL & set(ops)
    assert all(
        len(n.input) == 2 for n in exported.graph.node if n.op_type == "MatMul"
    )
    assert "Gelu" in ops and "Mul" not in ops  # folded scales / GELU
    assert exported.opset_import[0].version == 20
    np.testing.assert_allclose(_run(exported, x), _run(model, x), atol=1e-5)


def test_silu_is_expanded_back_to_standard_ops():
    model = build_yolo_like(size=32)
    x = (
        np.random.default_rng(0)
        .uniform(0, 1, (2, 3, 32, 32))
        .astype(np.float32)
    )
    exported = OnnxGraph.from_model(model).to_model()
    ops = {n.op_type for n in exported.graph.node}
    assert "Silu" not in ops and {"Sigmoid", "Mul"} <= ops
    np.testing.assert_allclose(_run(exported, x), _run(model, x), atol=1e-5)


def test_reducemean_axes_become_an_input_for_opset_18_plus():
    b = ModelBuilder(0)
    h = b.op("ReduceMean", ["images"], axes=[1], keepdims=1)
    erf = b.op("Erf", [b.op("Div", [h, b.const(np.float32(np.sqrt(2.0)))])])
    y = b.op(
        "Mul",
        [
            b.op("Mul", [h, b.const(np.float32(0.5))]),
            b.op("Add", [erf, b.const(np.float32(1.0))]),
        ],
    )
    model = b.build("images", [2, 6, 4], y, [2, 1, 4])
    exported = OnnxGraph.from_model(model).to_model()
    assert exported.opset_import[0].version == 20  # Gelu needs 20
    rm = next(n for n in exported.graph.node if n.op_type == "ReduceMean")
    assert len(rm.input) == 2
    x = np.random.default_rng(0).standard_normal((2, 6, 4)).astype(np.float32)
    np.testing.assert_allclose(_run(exported, x), _run(model, x), atol=1e-5)


def test_old_opsets_are_rejected():
    b = ModelBuilder(0, opset=11)
    y = b.op("Relu", ["images"])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)]
    )
    model.ir_version = 8
    with pytest.raises(NotImplementedError, match="opset"):
        OnnxGraph.from_model(model).to_model()


def test_saved_file_quantizes_like_the_original(tmp_path):
    model = build_vit(batch=2, depth=1, ln_style="op", gelu_style="op")
    path = tmp_path / "opt.onnx"
    OnnxGraph.from_model(model).save(path)
    rng = np.random.default_rng(0)
    cal = [
        rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
        for _ in range(2)
    ]
    a, b = QuantContainer(model, "int16"), QuantContainer(str(path), "int16")
    a.calibrate(cal)
    b.calibrate(cal)
    np.testing.assert_array_equal(
        a.run_graph(cal[0], "integer"), b.run_graph(cal[0], "integer")
    )
    again = OnnxGraph.from_model(str(path))
    assert {k: v for k, v in again.fusions.items() if v} == {"matmul_bias": 4}


def test_unused_initializers_are_dropped(tmp_path):
    model = build_vit(batch=2, depth=1)  # decomposed LN leaves constants
    exported = OnnxGraph.from_model(model).to_model()
    used = {i for n in exported.graph.node for i in n.input}
    assert {t.name for t in exported.graph.initializer} <= used


def test_optimize_onnx_cli(tmp_path, capsys):
    src, dst = tmp_path / "vit.onnx", tmp_path / "vit_opt.onnx"
    onnx.save(build_vit(batch=1, depth=2), str(src))
    out = optimize_onnx.main([str(src), "-o", str(dst), "--check"])
    assert out == str(dst) and dst.exists()
    text = capsys.readouterr().out
    assert "nodes ->" in text and "max |original - optimized|" in text
    n0, _ = optimize_onnx.summarize(str(src))
    n1, ops = optimize_onnx.summarize(str(dst))
    assert n1 < n0 and "LayerNormalization" in ops
