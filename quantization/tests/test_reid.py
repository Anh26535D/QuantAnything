import json

import numpy as np
import onnx
import pytest
from PIL import Image

import reid_zero_shot
from quantization import QuantContainer, reid
from quantization.tests.models import build_vit


def test_parse_name():
    assert reid.parse_name("/x/0002_c1s1_000451_03.jpg") == (2, 1)
    assert reid.parse_name("-1_c3s2_000001_00.jpg") == (-1, 3)
    with pytest.raises(ValueError):
        reid.parse_name("cat.jpg")


def test_ranking_protocol_excludes_same_camera_and_junk():
    #            g0     g1     g2     g3     g4
    dist = np.array([[0.3, 0.1, 0.4, 0.2, 0.5]])
    q_pid, q_cam = np.array([1]), np.array([1])
    g_pid = np.array([1, 2, 1, -1, 1])
    g_cam = np.array([1, 2, 2, 1, 3])
    res = reid.evaluate_ranking(dist, q_pid, g_pid, q_cam, g_cam)
    # valid ranking: g1(no) g2(yes) g4(yes) -> AP = (1/2 + 2/3) / 2
    assert res["mAP"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert res["rank1"] == 0.0 and res["rank5"] == 1.0
    assert res["num_valid_queries"] == 1


def test_ranking_perfect_and_skips_unmatched_queries():
    dist = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    res = reid.evaluate_ranking(
        dist,
        np.array([1, 2, 9]),
        np.array([1, 2]),
        np.array([1, 1, 1]),
        np.array([2, 2]),
    )
    assert res["mAP"] == 1.0 and res["rank1"] == 1.0
    assert res["num_valid_queries"] == 2  # pid 9 has no match
    with pytest.raises(ValueError, match="no query"):
        reid.evaluate_ranking(
            dist[2:],
            np.array([9]),
            np.array([1, 2]),
            np.array([1, 1]),
            np.array([2, 2]),
        )


def test_cosine_distance_is_scale_invariant():
    a = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
    np.testing.assert_allclose(
        reid.cosine_distance(a, a * 7), reid.cosine_distance(a, a), atol=1e-6
    )
    assert np.allclose(np.diag(reid.cosine_distance(a, a)), 0, atol=1e-6)


def _make_market(root, ids=6, cams=3, per=2, seed=0):
    """Tiny Market-1501 look-alike: identity = a colour-block pattern."""
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, 255, (ids, 4, 2, 3)).astype(np.uint8)

    def write(folder, pid, cam, idx):
        base = np.kron(patterns[pid - 1], np.ones((16, 16, 1), np.uint8))
        noisy = np.clip(
            base.astype(int) + rng.integers(-12, 12, base.shape) + 5 * cam,
            0,
            255,
        ).astype(np.uint8)
        name = f"{pid:04d}_c{cam}s1_{idx:06d}_00.jpg"
        Image.fromarray(noisy).save(root / folder / name)

    for folder in ("query", "bounding_box_test"):
        (root / folder).mkdir(parents=True)
    for pid in range(1, ids + 1):
        for cam in range(1, cams + 1):
            write("query", pid, cam, 0)
            for k in range(per):
                write("bounding_box_test", pid, cam, k + 1)
    # junk and distractor images in the gallery
    Image.fromarray(patterns[0].repeat(16, 0).repeat(16, 1)).save(
        root / "bounding_box_test" / "-1_c1s1_000001_00.jpg"
    )
    Image.fromarray(patterns[1].repeat(16, 0).repeat(16, 1)).save(
        root / "bounding_box_test" / "0000_c2s1_000001_00.jpg"
    )
    return root


def test_load_split_and_subsampling(tmp_path):
    root = _make_market(tmp_path / "Market")
    q, g = reid.load_split(str(tmp_path))  # root found one level down
    assert len(q[0]) == 18 and len(g[0]) == 6 * 3 * 2 + 2
    assert set(g[1]) >= {-1, 0}
    q2, g2 = reid.load_split(str(root), max_ids=2, distractors=3)
    assert len(np.unique(q2[1])) == 2
    keep = set(q2[1])
    assert sum(p in keep for p in g2[1]) == 2 * 3 * 2
    assert len(g2[0]) == 2 * 3 * 2 + 3


def test_embedding_model_cuts_the_classifier_head():
    model = build_vit(img=32, classes=10)
    with_head = QuantContainer(model).run_graph(
        np.zeros((1, 3, 32, 32), np.float32), "float"
    )
    assert with_head.shape == (1, 10)
    cut, tensor = reid.embedding_model(build_vit(img=32, dim=32, classes=10))
    emb = QuantContainer(cut).run_graph(np.zeros((1, 3, 32, 32)), "float")
    assert emb.shape == (1, 32)
    assert "Gemm" not in {n.op_type for n in cut.graph.node}
    onnx.checker.check_model(cut)
    with pytest.raises(ValueError, match="not found"):
        reid.embedding_model(build_vit(img=32), "nope")


def test_extract_features_pads_last_batch(tmp_path):
    root = _make_market(tmp_path / "m")
    paths = reid.list_images(str(root / "query"))[:5]
    seen = []

    def infer(x):
        seen.append(x.shape)
        return x.mean(axis=(2, 3))

    feats = reid.extract_features(
        infer, paths, 2, (32, 32), (0,) * 3, (1,) * 3
    )
    assert feats.shape == (5, 3) and seen == [(2, 3, 32, 32)] * 3


def test_cli_end_to_end_on_synthetic_dataset(tmp_path):
    _make_market(tmp_path / "Market")
    onnx_path = tmp_path / "vit.onnx"
    onnx.save(build_vit(img=32, patch=8, dim=32, classes=10), str(onnx_path))
    out = tmp_path / "res.json"
    results = reid_zero_shot.main(
        [
            "--onnx",
            str(onnx_path),
            "--root",
            str(tmp_path / "Market"),
            "--quant-types",
            "int16",
            "int8",
            "--modes",
            "integer",
            "--calib-images",
            "8",
            "--norm",
            "none",
            "--json",
            str(out),
        ]
    )
    assert json.loads(out.read_text()) == results
    assert set(results) == {"float", "int16 / integer", "int8 / integer"}
    # a random ViT already separates the synthetic identities
    assert results["float"]["mAP"] > 0.5
    # int16 simulates the float model almost exactly, int8 stays usable
    assert (
        abs(results["int16 / integer"]["mAP"] - results["float"]["mAP"]) < 0.05
    )
    assert results["int8 / integer"]["mAP"] > 0.3
