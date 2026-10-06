"""End-to-end behaviour of QuantContainer on a miniature YOLOv8-cls graph."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from quantization import QuantContainer, native
from quantization.tests.models import ModelBuilder, build_yolo_like


@pytest.fixture(scope="module")
def model():
    return build_yolo_like(seed=0)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    return [
        rng.uniform(0, 1, (4, 3, 32, 32)).astype(np.float32) for _ in range(3)
    ]


def _ort_ref(model, x):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {"images": x})[0]


def test_float_mode_matches_onnxruntime(model, data):
    c = QuantContainer(model)
    np.testing.assert_allclose(
        c.run_graph(data[0], "float"), _ort_ref(model, data[0]), atol=1e-6
    )


@pytest.mark.parametrize(
    "quant_type,tol", [("int8", 0.05), ("uint8", 0.05), ("int16", 5e-4)]
)
def test_quantized_output_tracks_float_model(model, data, quant_type, tol):
    c = QuantContainer(model, quant_type)
    c.calibrate(data)
    ref = _ort_ref(model, data[0])
    for mode in ("quantize", "integer"):
        out = c.quantized_infer(data[0], mode=mode)
        assert out.shape == ref.shape and out.dtype == np.float32
        assert np.abs(out - ref).max() < tol, mode
    # integer-only math agrees with the fake-quant simulation
    fake = c.quantized_infer(data[0], mode="quantize")
    integer = c.quantized_infer(data[0], mode="integer")
    assert np.abs(fake - integer).max() < max(tol, 1e-3)


def test_more_bits_means_less_error(model, data):
    ref = _ort_ref(model, data[0])
    errs = {}
    for qt in ("int8", "int16"):
        c = QuantContainer(model, qt)
        c.calibrate(data)
        errs[qt] = np.abs(c.run_graph(data[0], "integer") - ref).max()
    assert errs["int16"] < errs["int8"] / 10


def test_native_and_numpy_backends_are_bit_exact(model, data, monkeypatch):
    if not native.available():
        pytest.skip("native kernels missing")
    c = QuantContainer(model, "int8")
    c.calibrate(data)
    fast = c.run_graph(data[0], "integer")
    monkeypatch.setattr(native, "_lib", None)
    slow = c.run_graph(data[0], "integer")
    np.testing.assert_array_equal(fast, slow)


def test_calibration_accepts_generators_arrays_and_dicts(model, data):
    ref = QuantContainer(model)
    ref.calibrate(data)

    gen = QuantContainer(model)
    gen.calibrate(b for b in data)
    dct = QuantContainer(model)
    dct.calibrate([{"images": b} for b in data])
    for other in (gen, dct):
        assert other.qparams == ref.qparams

    single = QuantContainer(model)
    single.calibrate(data[0])
    assert single.calibrated


def test_calibration_requires_data(model):
    with pytest.raises(ValueError, match="no calibration data"):
        QuantContainer(model).calibrate([])


def test_quantized_inference_requires_calibration(model, data):
    with pytest.raises(ValueError, match="not calibrated"):
        QuantContainer(model).run_graph(data[0], "integer")


def test_unknown_mode_is_rejected(model, data):
    c = QuantContainer(model)
    c.calibrate(data)
    with pytest.raises(ValueError, match="Unknown graph mode"):
        c.run_graph(data[0], "bogus")


def test_scale_inheritance_between_layers(model, data):
    c = QuantContainer(model)
    c.calibrate(data)
    for layer in c.layers:
        for name, qp in zip(layer.data_inputs, layer.in_qp):
            assert c.qparams[name] == qp
        for name, qp in zip(layer.node.outputs, layer.out_qp):
            assert c.qparams[name] == qp


def test_compare_layers_reports_every_layer(model, data):
    c = QuantContainer(model)
    c.calibrate(data)
    for mode in ("quantize", "integer"):
        report = c.compare_layers(data[0], mode=mode)
        assert list(report) == [n.name for n in c.graph.nodes]
        last = report[c.graph.nodes[-1].name]
        assert last["layer_type"] == "Softmax"
        assert 0 <= last["mean_diff"] <= last["max_diff"] < 0.05
    with pytest.raises(ValueError):
        c.compare_layers(data[0], mode="float")


def test_save_and_load_roundtrip(model, data, tmp_path):
    c = QuantContainer(model, "int16")
    c.calibrate(data)
    path = tmp_path / "qparams.json"
    c.save(path)
    loaded = QuantContainer.load(model, path)
    assert loaded.quant_type == "int16" and loaded.calibrated
    for mode in ("quantize", "integer"):
        np.testing.assert_array_equal(
            c.run_graph(data[1], mode), loaded.run_graph(data[1], mode)
        )


def test_load_rejects_mismatching_model(model, data, tmp_path):
    c = QuantContainer(model)
    c.calibrate(data)
    path = tmp_path / "q.json"
    c.save(path)
    b = ModelBuilder(0)
    y = b.op("Relu", ["other"])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [helper.make_tensor_value_info("other", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info(y, TensorProto.FLOAT, [1])],
        b.inits,
    )
    other = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    other.ir_version = 8
    with pytest.raises(ValueError, match="do not match"):
        QuantContainer.load(other, path)


def test_multiple_inputs_and_outputs():
    b = ModelBuilder(0)
    s = b.op("Add", ["a", "b"])
    t = b.op("Relu", [s])
    graph = helper.make_graph(
        b.nodes,
        "g",
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 3])
            for n in "ab"
        ],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 3])
            for n in (s, t)
        ],
        b.inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    rng = np.random.default_rng(0)
    feeds = {
        "a": rng.standard_normal((2, 3)).astype(np.float32),
        "b": rng.standard_normal((2, 3)).astype(np.float32),
    }
    c = QuantContainer(m)
    with pytest.raises(ValueError, match="dict"):
        c.run_graph(feeds["a"], "float")
    c.calibrate(feeds)
    out_sum, out_relu = c.run_graph(feeds, "integer")
    np.testing.assert_allclose(out_sum, feeds["a"] + feeds["b"], atol=0.05)
    np.testing.assert_allclose(
        out_relu, np.maximum(feeds["a"] + feeds["b"], 0), atol=0.05
    )


def test_batch_size_can_change_after_calibration(model, data):
    c = QuantContainer(model)
    c.calibrate(data)
    one = c.run_graph(data[0][:1], "integer")
    many = c.run_graph(np.concatenate(data), "integer")
    np.testing.assert_array_equal(one, many[:1])
