"""vit_from_npz.py on random weights with the flax checkpoint layout."""

import numpy as np
import onnxruntime as ort
import pytest

import vit_from_npz
from quantization import OnnxGraph, QuantContainer

DIM, HEADS, DEPTH, PATCH, IMG, CLASSES, MLP = 16, 2, 2, 8, 16, 5, 32


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    rng = np.random.default_rng(0)
    r = lambda *s: (0.2 * rng.standard_normal(s)).astype(np.float32)  # noqa: E731
    hd, tokens = DIM // HEADS, (IMG // PATCH) ** 2 + 1
    w = {
        "embedding/kernel": r(PATCH, PATCH, 3, DIM),
        "embedding/bias": r(DIM),
        "cls": r(1, 1, DIM),
        "Transformer/posembed_input/pos_embedding": r(1, tokens, DIM),
        "Transformer/encoder_norm/scale": 1 + r(DIM),
        "Transformer/encoder_norm/bias": r(DIM),
        "head/kernel": r(DIM, CLASSES),
        "head/bias": r(CLASSES),
    }
    for i in range(DEPTH):
        p = f"Transformer/encoderblock_{i}/"
        a = p + "MultiHeadDotProductAttention_1/"
        for ln in ("LayerNorm_0", "LayerNorm_2"):
            w[p + ln + "/scale"], w[p + ln + "/bias"] = 1 + r(DIM), r(DIM)
        for n in ("query", "key", "value"):
            w[a + n + "/kernel"], w[a + n + "/bias"] = (
                r(DIM, HEADS, hd),
                r(HEADS, hd),
            )
        w[a + "out/kernel"], w[a + "out/bias"] = r(HEADS, hd, DIM), r(DIM)
        m = p + "MlpBlock_3/"
        w[m + "Dense_0/kernel"], w[m + "Dense_0/bias"] = r(DIM, MLP), r(MLP)
        w[m + "Dense_1/kernel"], w[m + "Dense_1/bias"] = r(MLP, DIM), r(DIM)
    path = tmp_path_factory.mktemp("w") / "tiny.npz"
    np.savez(path, **w)
    return np.load(path), w


def _reference(w, x):
    """Straight NumPy transcription of the flax ViT forward pass."""
    n = x.shape[0]
    ker = w["embedding/kernel"]
    gh = IMG // PATCH
    patches = x.reshape(n, 3, gh, PATCH, gh, PATCH).transpose(0, 2, 4, 3, 5, 1)
    tok = (
        patches.reshape(n, gh * gh, -1) @ ker.reshape(-1, DIM)
        + w["embedding/bias"]
    )
    h = np.concatenate([np.tile(w["cls"], (n, 1, 1)), tok], 1)
    h = h + w["Transformer/posembed_input/pos_embedding"]

    def ln(v, p):
        mu, var = v.mean(-1, keepdims=True), v.var(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + 1e-6) * w[p + "/scale"] + w[
            p + "/bias"
        ]

    def gelu(v):
        return (
            0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3)))
        )

    hd = DIM // HEADS
    for i in range(DEPTH):
        p = f"Transformer/encoderblock_{i}/"
        a = p + "MultiHeadDotProductAttention_1/"
        y = ln(h, p + "LayerNorm_0")
        q, k, v = [
            np.einsum("btd,dhe->bhte", y, w[a + s + "/kernel"])
            + w[a + s + "/bias"][None, :, None, :]
            for s in ("query", "key", "value")
        ]
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (
            np.einsum("bhte,hed->btd", att @ v, w[a + "out/kernel"])
            + w[a + "out/bias"]
        )
        h = h + o
        y = ln(h, p + "LayerNorm_2")
        m = p + "MlpBlock_3/"
        y = gelu(y @ w[m + "Dense_0/kernel"] + w[m + "Dense_0/bias"])
        h = h + y @ w[m + "Dense_1/kernel"] + w[m + "Dense_1/bias"]
    cls = ln(h, "Transformer/encoder_norm")[:, 0]
    return cls, cls @ w["head/kernel"] + w["head/bias"]


@pytest.mark.parametrize("embedding", [False, True])
def test_onnx_matches_flax_math(weights, embedding):
    npz, w = weights
    model = vit_from_npz.build_onnx(npz, batch=2, embedding=embedding)
    x = (
        np.random.default_rng(1)
        .standard_normal((2, 3, IMG, IMG))
        .astype(np.float32)
    )
    feat, logits = _reference(w, x)
    expected = feat if embedding else logits
    got = ort.InferenceSession(model.SerializeToString()).run(
        None, {"images": x}
    )[0]
    np.testing.assert_allclose(got, expected, atol=2e-4)
    mine = QuantContainer(model).run_graph(x, "float")
    np.testing.assert_allclose(mine, expected, atol=2e-4)


def test_graph_is_quantizable(weights):
    npz, _ = weights
    g = OnnxGraph.from_model(vit_from_npz.build_onnx(npz))
    assert g.fusions["matmul_bias"] == 4 * DEPTH  # qkv, proj, fc1, fc2
    ops = {n.op_type for n in g.nodes}
    assert {"LayerNormalization", "Gelu", "Softmax"} <= ops


def test_variants_point_to_google_storage():
    assert set(vit_from_npz.VARIANTS) == {"ti16", "s16", "b16"}
    assert all(
        v.endswith("res_224.npz") for v in vit_from_npz.VARIANTS.values()
    )
