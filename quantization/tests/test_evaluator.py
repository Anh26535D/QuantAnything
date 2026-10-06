import numpy as np
import onnx
import onnxruntime as ort
import pytest
from PIL import Image

from quantization import QuantContainer, evaluator
from quantization.tests.models import build_yolo_like


@pytest.fixture
def onnx_file(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(build_yolo_like(size=224, classes=3), str(path))
    return str(path)


def _make_dataset(root, classes=3, per_class=2):
    rng = np.random.default_rng(0)
    for c in range(classes):
        (root / f"class{c}").mkdir(parents=True)
        for i in range(per_class):
            pixels = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / f"class{c}" / f"{i}.png")


def test_macro_metrics():
    acc, prec, rec, f1 = evaluator.macro_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert acc == 0.75
    assert prec == pytest.approx((1.0 + 2 / 3) / 2)
    assert rec == pytest.approx((0.5 + 1.0) / 2)
    assert f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert evaluator.macro_metrics([1, 2], [1, 2]) == (1.0, 1.0, 1.0, 1.0)


def test_load_dataset_reads_labelled_batches(tmp_path):
    _make_dataset(tmp_path)
    batches = list(evaluator.load_dataset(str(tmp_path), batch_size=4))
    xs = np.concatenate([b[0] for b in batches])
    ys = np.concatenate([b[1] for b in batches])
    assert xs.shape == (6, 3, 224, 224) and xs.dtype == np.float32
    assert 0.0 <= xs.min() and xs.max() <= 1.0
    np.testing.assert_array_equal(ys, [0, 0, 1, 1, 2, 2])


def test_load_dataset_falls_back_to_dummy_data(tmp_path):
    x, y = next(evaluator.load_dataset(str(tmp_path / "missing"), 2))
    assert x.shape == (2, 3, 224, 224) and y is None


def test_run_evaluation_end_to_end(onnx_file, tmp_path, capsys):
    dataset = tmp_path / "val"
    _make_dataset(dataset)
    result = evaluator.run_evaluation(
        onnx_file, str(dataset), "int8", "integer"
    )
    assert result["samples"] == 6
    assert result["onnx_vs_float"] < 1e-5
    assert result["float_vs_quant"] < 0.2
    assert "accuracy_quant" in result
    assert "Classification metrics" in capsys.readouterr().out


def test_evaluate_model_without_labels(onnx_file, tmp_path):
    container = QuantContainer(onnx_file)
    container.calibrate(next(evaluator.load_dataset("missing", 4))[0])
    result = evaluator.evaluate_model(
        ort.InferenceSession(onnx_file), container, str(tmp_path / "none")
    )
    assert result["samples"] == 3 and "accuracy_quant" not in result
