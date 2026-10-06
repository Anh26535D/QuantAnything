"""Zero-shot person ReID: float vs quantized ViT (or any ONNX backbone).

The embedding of a pretrained, *not* ReID-trained backbone is used as the
re-identification feature (cosine distance). The script reports mAP / CMC for
the float model (ONNX Runtime) and for every requested quantization setting,
so you can see what quantization costs on a public ReID benchmark.

Dataset (Market-1501, public): download it yourself (e.g. from the dataset's
official page, Kaggle or the Hugging Face Hub) and point ``--root`` at the
folder that contains ``query/`` and ``bounding_box_test/``.

Typical use:
    uv run python download_vit.py --source timm --model vit_tiny_patch16_224 \\
        --embedding --output vit_tiny.onnx
    uv run python reid_zero_shot.py --onnx vit_tiny.onnx \\
        --root Market-1501-v15.09.15 --max-ids 100 --distractors 1000 \\
        --quant-types int8 int16 --modes integer

A classification model (``vit.onnx`` from ``download_vit.py``) also works: the
head is cut automatically (``--feature-tensor`` overrides the choice).
The full Market-1501 test set has ~19.7k gallery images; the NumPy engine is a
simulator, so start with ``--max-ids`` / ``--distractors`` and scale up.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm

from quantization import QuantContainer, reid


def _model_geometry(model):
    """``(batch, (height, width))`` from the first graph input."""
    init = {t.name for t in model.graph.initializer}
    inp = next(i for i in model.graph.input if i.name not in init)
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or 0 in dims:
        raise SystemExit(
            f"input '{inp.name}' needs a static NCHW shape, got {dims}; "
            "re-export with a fixed batch (download_vit.py --batch N)"
        )
    return dims[0], (dims[2], dims[3])


def _score(feats_q, feats_g, q, g):
    dist = reid.cosine_distance(feats_q, feats_g)
    return reid.evaluate_ranking(dist, q[1], g[1], q[2], g[2])


def _row(name, res, cos=None):
    extra = "" if cos is None else f" | cos(float)={cos:.4f}"
    print(
        f"{name:<24} mAP={res['mAP'] * 100:6.2f}  "
        f"R1={res['rank1'] * 100:6.2f}  R5={res['rank5'] * 100:6.2f}  "
        f"R10={res['rank10'] * 100:6.2f}{extra}"
    )


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--onnx", required=True, help="backbone ONNX (static batch)"
    )
    p.add_argument("--root", required=True, help="Market-1501 style folder")
    p.add_argument("--feature-tensor", default=None)
    p.add_argument(
        "--norm", default="half", choices=sorted(reid.NORMALIZATION)
    )
    p.add_argument("--quant-types", nargs="*", default=["int8", "int16"])
    p.add_argument(
        "--modes",
        nargs="*",
        default=["integer"],
        choices=["quantize", "integer"],
    )
    p.add_argument("--calib-images", type=int, default=64)
    p.add_argument("--max-ids", type=int, default=None)
    p.add_argument("--distractors", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write results to this file")
    args = p.parse_args(argv)

    if not os.path.exists(args.onnx):
        sys.exit(f"{args.onnx} not found (see download_vit.py)")
    mean, std = reid.NORMALIZATION[args.norm]

    model, tensor = reid.embedding_model(
        onnx.load(args.onnx), args.feature_tensor
    )
    batch, size = _model_geometry(model)
    print(f"Feature tensor: {tensor} | batch {batch} | input {size}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "embedding.onnx")
        onnx.save(model, path)
        session = ort.InferenceSession(path)
        in_name = session.get_inputs()[0].name

        query, gallery = reid.load_split(
            args.root, args.max_ids, args.distractors, args.seed
        )
        print(
            f"Query {len(query[0])} images, gallery {len(gallery[0])} images, "
            f"{len(np.unique(query[1]))} identities"
        )

        def run(infer, label):
            fq = reid.extract_features(
                infer,
                query[0],
                batch,
                size,
                mean,
                std,
                lambda it: tqdm(it, desc=f"{label} query", leave=False),
            )
            fg = reid.extract_features(
                infer,
                gallery[0],
                batch,
                size,
                mean,
                std,
                lambda it: tqdm(it, desc=f"{label} gallery", leave=False),
            )
            return fq, fg

        results, feats = {}, {}
        float_infer = lambda x: session.run(None, {in_name: x})[0]  # noqa: E731
        feats["float"] = run(float_infer, "float")
        results["float"] = _score(*feats["float"], query, gallery)
        print("\nZero-shot ReID")
        _row("float (ONNX Runtime)", results["float"])

        rng = np.random.default_rng(args.seed)
        calib_paths = rng.choice(
            gallery[0], min(args.calib_images, len(gallery[0])), replace=False
        )
        calib = [
            np.stack(
                [
                    reid.load_image(f, size, mean, std)
                    for f in calib_paths[s : s + batch]
                ]
            )
            for s in range(0, len(calib_paths) - batch + 1, batch)
        ]
        if not calib:
            sys.exit("not enough images for one calibration batch")

        for qt in args.quant_types:
            container = QuantContainer(path, quant_type=qt)
            container.calibrate(calib)
            for mode in args.modes:
                name = f"{qt} / {mode}"
                infer = lambda x, m=mode: container.run_graph(x, m)  # noqa: E731
                fq, fg = run(infer, name)
                results[name] = _score(fq, fg, query, gallery)
                ref_q, ref_g = feats["float"]
                cos = float(
                    np.mean(
                        np.sum(
                            reid.l2_normalize(fg) * reid.l2_normalize(ref_g), 1
                        )
                    )
                )
                _row(name, results[name], cos)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    main()
