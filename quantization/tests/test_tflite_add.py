"""Bit-exactness of the TFLite Add port against vectors from Google's code."""

import os

import numpy as np
import pytest

from quantization import tflite_add as T

GOLDEN = os.path.join(
    os.path.dirname(__file__), "data", "tflite_add_golden.npz"
)


def test_golden_vectors():
    g = np.load(GOLDEN)
    meta = g["meta"]
    assert len(meta) >= 100
    for i, (kind, s1, s2, so, z1, z2, zo) in enumerate(meta):
        x1, x2, ref = g[f"x1_{i}"], g[f"x2_{i}"], g[f"ref_{i}"]
        if int(kind) == 2:
            out = T.add_pot(x1, x2, T.pot_shifts(s1, s2, so))
        else:
            lo, hi = (-128, 127) if int(kind) == 0 else (-32768, 32767)
            p = T.prepare(s1, s2, so, z1, z2, zo, lo, hi)
            out = T.add(x1, x2, p)
        np.testing.assert_array_equal(out, ref)


def test_srdhm_saturates():
    assert int(T.srdhm(T.INT32_MIN, T.INT32_MIN)) == T.INT32_MAX


def test_overflow_is_reported():
    p = T.prepare(0.01, 0.01, 0.02, 0, 0, 0, -32768, 32767, left_shift=20)
    with pytest.raises(OverflowError):
        T.add(np.array([30000]), np.array([0]), p)


def test_container_add_impl_matches_exact():
    from quantization import QuantContainer
    from quantization.tests.models import build_vit

    model = build_vit(batch=2, depth=2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
    outs = {}
    for impl in ("exact", "tflite"):
        c = QuantContainer(model, "int16", add_impl=impl)
        c.calibrate([x])
        outs[impl] = c.run_graph(x, "integer")
    a, b = outs["exact"].ravel(), outs["tflite"].ravel()
    cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    assert cos > 0.999
