"""Calibration and classification evaluation of a quantized ONNX model.

Only NumPy, ``onnxruntime`` (as the ground-truth reference) and Pillow are
needed. ``ultralytics`` is imported lazily and only to export the demo model.
"""

import argparse
import os

import numpy as np

from quantization.quant_container import QuantContainer

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def export_yolov8_cls(weights="yolov8n-cls.pt", imgsz=224):
    """Exports YOLOv8-cls weights to ONNX and returns the file path.

    Requires the optional ``ultralytics`` package.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is only needed to export the demo model; install "
            "it with `uv sync --extra export` or pass an existing .onnx"
        ) from exc
    return YOLO(weights).export(format="onnx", imgsz=imgsz)


def _dummy_batches(batch_size, count=3, size=224):
    rng = np.random.default_rng(0)
    for _ in range(count):
        yield (
            rng.uniform(0.0, 1.0, (batch_size, 3, size, size)).astype(
                np.float32
            ),
            None,
        )


def load_dataset(
    dataset_dir, batch_size=16, limit_per_class=2, limit_classes=5
):
    """Yields ``(images, labels)`` batches from ``dataset_dir/<class>/*.jpg``.

    Images are resized to 224x224 and returned as float32 NCHW in ``[0, 1]``.
    Falls back to random dummy batches (labels ``None``) if the directory is
    missing or empty. ``limit_per_class`` / ``limit_classes`` bound memory use.
    """
    from PIL import Image

    if not os.path.isdir(dataset_dir):
        print(
            f"Dataset directory '{dataset_dir}' not found. Using dummy inputs."
        )
        yield from _dummy_batches(batch_size)
        return

    class_names = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    if limit_classes is not None:
        class_names = class_names[:limit_classes]

    image_paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        files = sorted(
            f
            for f in os.listdir(class_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        for fname in files[:limit_per_class]:
            image_paths.append(os.path.join(class_path, fname))
            labels.append(idx)

    if not image_paths:
        print(f"No valid images found in '{dataset_dir}'. Using dummy inputs.")
        yield from _dummy_batches(batch_size)
        return

    print(
        f"Subsampled dataset loaded: {len(image_paths)} images across "
        f"{len(class_names)} classes (limit={limit_per_class} per class)."
    )

    for start in range(0, len(image_paths), batch_size):
        batch, batch_labels = [], []
        for path, label in zip(
            image_paths[start : start + batch_size],
            labels[start : start + batch_size],
        ):
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
            except OSError as exc:
                print(f"Failed to load image {path}: {exc}")
                continue
            arr = np.asarray(img, dtype=np.float32) / 255.0
            batch.append(arr.transpose(2, 0, 1))
            batch_labels.append(label)
        if batch:
            yield np.stack(batch), np.array(batch_labels)


def macro_metrics(y_true, y_pred):
    """Accuracy and macro-averaged precision / recall / F1.

    Returns:
        ``(accuracy, precision, recall, f1)`` as floats in ``[0, 1]``.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        pred_n, true_n = float(np.sum(y_pred == c)), float(np.sum(y_true == c))
        p = tp / pred_n if pred_n else 0.0
        r = tp / true_n if true_n else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return (
        float(np.mean(y_true == y_pred)),
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


def evaluate_model(
    ort_session, quant_graph, dataset_dir, batch_size=1, infer_mode="quantize"
):
    """Compares ONNX Runtime, the NumPy float graph and the quantized graph.

    Prints per-batch and aggregate errors (and classification metrics when the
    dataset has labels) and returns the aggregate numbers as a dict.
    """
    from tqdm import tqdm

    input_name = ort_session.get_inputs()[0].name
    onnx_float_errors, float_quant_errors, kl_divs = [], [], []
    labels, preds = [], {"onnx": [], "float": [], "quant": []}
    total = 0

    batches = tqdm(
        load_dataset(dataset_dir, batch_size=batch_size),
        desc="Evaluating batches",
        unit="batch",
    )
    for idx, (x, y) in enumerate(batches):
        if y is not None:
            labels.extend(y)
        out_onnx = ort_session.run(None, {input_name: x})[0]
        out_float = quant_graph.run_graph(x, mode="float")
        out_quant = quant_graph.quantized_infer(x, mode=infer_mode)

        e_onnx = float(np.max(np.abs(out_onnx - out_float)))
        e_quant = float(np.max(np.abs(out_float - out_quant)))
        onnx_float_errors.append(e_onnx)
        float_quant_errors.append(e_quant)

        eps = 1e-15
        p, q = np.clip(out_float, eps, 1.0), np.clip(out_quant, eps, 1.0)
        kl = np.sum(p * np.log(p / q), axis=-1)
        kl_divs.extend(kl)

        total += x.shape[0]
        preds["onnx"].extend(np.argmax(out_onnx, axis=1))
        preds["float"].extend(np.argmax(out_float, axis=1))
        preds["quant"].extend(np.argmax(out_quant, axis=1))
        batches.write(
            f"Batch {idx + 1}: max |ONNX - float| = {e_onnx:.6f} | "
            f"max |float - quantized| = {e_quant:.6f} | "
            f"KL = {np.mean(kl):.6f}"
        )

    result = {
        "samples": total,
        "onnx_vs_float": float(np.mean(onnx_float_errors)),
        "float_vs_quant": float(np.mean(float_quant_errors)),
        "kl_divergence": float(np.mean(kl_divs)),
    }
    print("\n================== Evaluation Results ==================")
    print(f"Total samples evaluated: {total}")
    print(
        "Average ONNX Runtime vs NumPy float graph (max abs diff): "
        f"{result['onnx_vs_float']:.6f}"
    )
    print(
        "Average float vs quantized (max abs diff): "
        f"{result['float_vs_quant']:.6f}"
    )
    print(f"Average KL(float || quantized): {result['kl_divergence']:.6f}")

    if labels:
        tag = (
            f"Quantized ({quant_graph.quant_type.upper()}, "
            f"{infer_mode.upper()})"
        )
        rows = [
            ("ONNX", "onnx", 0.0),
            ("NumPy float", "float", 0.0),
            (tag, "quant", result["kl_divergence"]),
        ]
        print("\nClassification metrics (macro-averaged):")
        print(
            f"{'Model':<28} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | "
            f"{'F1':<8} | {'KL':<10}"
        )
        print("-" * 82)
        for name, key, kl in rows:
            acc, prec, rec, f1 = macro_metrics(labels, preds[key])
            result[f"accuracy_{key}"] = acc
            print(
                f"{name:<28} | {acc * 100:<7.2f}% | {prec * 100:<7.2f}% | "
                f"{rec * 100:<7.2f}% | {f1 * 100:<7.2f}% | {kl:<10.6f}"
            )
    else:
        print(
            "Note: no labels available (dummy input mode), classification "
            "metrics were skipped."
        )
    print("========================================================")
    return result


def run_evaluation(
    onnx_path="yolov8n-cls.onnx",
    dataset_dir="dataset",
    quant_type="int8",
    infer_mode="quantize",
):
    """Calibrates and evaluates a classification model end to end."""
    import onnxruntime as ort

    print("\n==================================================")
    print(f"Quantization assessment ({quant_type}, {infer_mode})")
    print("==================================================")

    if not os.path.exists(onnx_path):
        print(f"\n--- '{onnx_path}' not found, exporting with ultralytics ---")
        onnx_path = export_yolov8_cls()

    print("\n--- Loading ONNX model ---")
    quant_graph = QuantContainer(onnx_path, quant_type=quant_type)
    ort_session = ort.InferenceSession(onnx_path)

    print("\n--- Calibrating ---")
    if os.path.isdir(dataset_dir):
        calib = [
            x
            for x, _ in load_dataset(
                dataset_dir, batch_size=4, limit_per_class=2, limit_classes=5
            )
        ]
    else:
        print("Dataset directory not found. Calibrating on random inputs...")
        calib = [x for x, _ in _dummy_batches(8)]
    quant_graph.calibrate(calib)

    print("\n--- Evaluating ---")
    return evaluate_model(
        ort_session,
        quant_graph,
        dataset_dir,
        batch_size=1,
        infer_mode=infer_mode,
    )


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="yolov8n-cls.onnx")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--quant-type", default="int8")
    parser.add_argument(
        "--mode", default="quantize", choices=["quantize", "integer"]
    )
    args = parser.parse_args(argv)
    run_evaluation(args.onnx, args.dataset, args.quant_type, args.mode)


if __name__ == "__main__":
    main()
