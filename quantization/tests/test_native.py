"""The C++ kernels must agree exactly with the NumPy fallbacks."""

import numpy as np
import pytest

from quantization import native

CONV_CASES = [
    # N, C, H, W, Cout, K, stride, pad, dilation, groups
    (2, 6, 9, 11, 8, 3, 1, 1, 1, 1),
    (1, 8, 16, 16, 8, 3, 2, 1, 1, 8),  # depthwise, strided
    (2, 4, 10, 10, 6, 1, 1, 0, 1, 2),  # pointwise, grouped
    (1, 3, 12, 12, 4, 3, 1, 2, 2, 1),  # dilated
    (3, 5, 7, 7, 5, 5, 1, 0, 1, 1),  # no padding, big kernel
]


def _reference_conv(x, w, strides, pads, dil, groups):
    return native._np_conv(
        x.astype(np.int64),
        w.astype(np.int64),
        strides,
        pads,
        dil,
        groups,
        np.int64,
    )


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("case", CONV_CASES)
def test_conv2d_int_matches_reference(backend, case, wide):
    n, c, h, w_, co, k, s, p, d, g = case
    rng = np.random.default_rng(1)
    x = rng.integers(-255, 256, (n, c, h, w_)).astype(np.int32)
    w = rng.integers(-128, 128, (co, c // g, k, k)).astype(np.int32)
    args = ((s, s), (p, p, p, p), (d, d), g)
    got = native.conv2d_int(x, w, *args, wide=wide)
    assert got.dtype == (np.int64 if wide else np.int32)
    np.testing.assert_array_equal(got, _reference_conv(x, w, *args))


@pytest.mark.parametrize("case", CONV_CASES)
def test_conv2d_f32_matches_reference(backend, case):
    n, c, h, w_, co, k, s, p, d, g = case
    rng = np.random.default_rng(2)
    x = rng.standard_normal((n, c, h, w_)).astype(np.float32)
    w = rng.standard_normal((co, c // g, k, k)).astype(np.float32)
    args = ((s, s), (p, p, p, p), (d, d), g)
    got = native.conv2d_f32(x, w, *args)
    ref = native._np_conv(x, w, *args, np.float32)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_conv2d_asymmetric_padding(backend):
    rng = np.random.default_rng(3)
    x = rng.integers(-9, 9, (1, 2, 6, 6)).astype(np.int32)
    w = rng.integers(-9, 9, (3, 2, 3, 3)).astype(np.int32)
    args = ((1, 2), (0, 2, 1, 0), (1, 1), 1)
    got = native.conv2d_int(x, w, *args, wide=True)
    np.testing.assert_array_equal(got, _reference_conv(x, w, *args))


@pytest.mark.parametrize("wide", [False, True])
def test_gemm_int(backend, wide):
    rng = np.random.default_rng(4)
    a = rng.integers(-255, 256, (7, 37)).astype(np.int32)
    b = rng.integers(-128, 128, (37, 530)).astype(np.int32)
    np.testing.assert_array_equal(
        native.gemm_int(a, b, wide), a.astype(np.int64) @ b.astype(np.int64)
    )


def test_gemm_f32(backend):
    rng = np.random.default_rng(5)
    a = rng.standard_normal((9, 33)).astype(np.float32)
    b = rng.standard_normal((33, 21)).astype(np.float32)
    np.testing.assert_allclose(
        native.gemm_f32(a, b), a @ b, rtol=1e-4, atol=1e-4
    )


def test_accumulator_width_selection():
    assert not native.accumulator_is_wide(8, 3 * 3 * 256)
    assert native.accumulator_is_wide(8, 1 << 20)
    assert native.accumulator_is_wide(16, 1)


@pytest.mark.skipif(not native.available(), reason="native kernels missing")
def test_native_matches_numpy_fallback_for_requantize(monkeypatch):
    rng = np.random.default_rng(6)
    acc = rng.integers(-(2**40), 2**40, (2, 5, 7)).astype(np.int64)
    mult = rng.integers(-(2**31) + 1, 2**31, 5)
    shift = rng.integers(-3, 25, 5)
    pre, post = rng.integers(-50, 50, 5), rng.integers(-50, 50, 5)
    args = (acc, 2, 5, 7, mult, shift, 3, -32768, 32767)
    fast = native.requantize(*args, pre_bias=pre, post_bias=post)
    monkeypatch.setattr(native, "_lib", None)
    slow = native.requantize(*args, pre_bias=pre, post_bias=post)
    np.testing.assert_array_equal(fast, slow)
    assert fast.dtype == np.int32


def test_requantize_int32_accumulator(backend):
    acc = np.arange(-60, 60, dtype=np.int32).reshape(2, 3, 20)
    m, s = 1 << 30, 0  # real multiplier = 2**30 * 2**-31 = 0.5
    out = native.requantize(acc, 2, 3, 20, m, s, 1, -128, 127)
    np.testing.assert_array_equal(
        out, np.clip(np.floor(acc * 0.5 + 0.5) + 1, -128, 127)
    )


def test_elementwise_kernels_match_fallback(monkeypatch):
    if not native.available():
        pytest.skip("native kernels missing")
    rng = np.random.default_rng(7)
    x = rng.integers(-1000, 1000, 1000).astype(np.int32)
    y = x[::-1].copy()
    table = np.arange(100, dtype=np.int32)
    calls = [
        lambda: native.rescale(x, 3, 1 << 30, 2, 1, -128, 127),
        lambda: native.add(
            x, y, 0, 1 << 30, 1, 2, 1 << 30, 3, 0, -(2**15), 2**15 - 1, -1
        ),
        lambda: native.mul(x, y, 1, 2, 1 << 30, 10, 0, -128, 127),
        lambda: native.lut(x, table, -50),
    ]
    fast = [c() for c in calls]
    monkeypatch.setattr(native, "_lib", None)
    for f, c in zip(fast, calls):
        np.testing.assert_array_equal(f, c())


def test_add_broadcasts(backend):
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.array([10], dtype=np.int32)
    one = (1 << 30, -1)  # real multiplier exactly 1.0
    out = native.add(a, b, 0, *one, 0, *one, 0, -128, 127)
    np.testing.assert_array_equal(out, a + b)
