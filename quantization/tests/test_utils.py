import numpy as np
import pytest

from quantization.utils import (
    QParams,
    calculate_scale_zp,
    dequantize,
    fake_quantize,
    multiply_by_quantized_multiplier,
    parse_quant_type,
    quantize,
    quantize_multiplier,
)


def test_parse_quant_type():
    assert parse_quant_type("int8") == (8, True)
    assert parse_quant_type("uint8") == (8, False)
    assert parse_quant_type("int9") == (9, True)
    assert parse_quant_type("uint12") == (12, False)
    with pytest.raises(ValueError):
        parse_quant_type("float32")


def test_scale_zp_int8_is_symmetric():
    scale, zp, qmin, qmax = calculate_scale_zp(-10.0, 10.0, "int8")
    assert (zp, qmin, qmax) == (0, -128, 127)
    assert scale == pytest.approx(10.0 / 127.0)


def test_scale_zp_uint8_is_asymmetric():
    scale, zp, qmin, qmax = calculate_scale_zp(-2.0, 8.0, "uint8")
    assert (qmin, qmax) == (0, 255)
    assert scale == pytest.approx(10.0 / 255.0)
    assert zp == int(np.round(2.0 / scale))


def test_scale_zp_uint_range_contains_zero():
    scale, zp, _, _ = calculate_scale_zp(2.0, 10.0, "uint8")
    assert zp == 0 and scale == pytest.approx(10.0 / 255.0)


def test_scale_zp_int9():
    scale, zp, qmin, qmax = calculate_scale_zp(-100.0, 50.0, "int9")
    assert (zp, qmin, qmax) == (0, -256, 255)
    assert scale == pytest.approx(100.0 / 255.0)


def test_scale_zp_degenerate_range():
    assert calculate_scale_zp(0.0, 0.0, "int8")[:2] == (1.0, 0)
    assert calculate_scale_zp(0.0, 0.0, "uint8")[:2] == (1.0, 0)


@pytest.mark.parametrize("quant_type", ["int8", "uint8", "int16"])
def test_quantize_roundtrip_error_below_half_step(backend, quant_type):
    x = np.random.default_rng(0).uniform(-3, 5, 1000).astype(np.float32)
    scale, zp, qmin, qmax = calculate_scale_zp(x.min(), x.max(), quant_type)
    q = quantize(x, scale, zp, qmin, qmax)
    assert q.dtype == np.int32 and q.min() >= qmin and q.max() <= qmax
    err = np.abs(dequantize(q, scale, zp) - x)
    assert err.max() <= scale / 2 * 1.001
    np.testing.assert_array_equal(
        dequantize(q, scale, zp), fake_quantize(x, scale, zp, qmin, qmax)
    )


def test_quantize_rounds_half_to_even(backend):
    q = quantize(
        np.array([0.5, 1.5, 2.5, -0.5, -1.5], np.float32), 1.0, 0, -128, 127
    )
    np.testing.assert_array_equal(q, [0, 2, 2, 0, -2])


def test_quantize_per_channel_scale():
    x = np.ones((2, 3), np.float32)
    q = quantize(x, np.array([[0.5], [0.25]]), np.array([[0], [0]]), -128, 127)
    np.testing.assert_array_equal(q, [[2] * 3, [4] * 3])


def test_quantize_saturates(backend):
    q = quantize(np.array([1e6, -1e6], np.float32), 0.1, 0, -128, 127)
    np.testing.assert_array_equal(q, [127, -128])


@pytest.mark.parametrize("real", [1e-5, 0.0123, 0.5, 1.0, 3.7, 250.0])
def test_quantize_multiplier_matches_real_value(real):
    m, s = quantize_multiplier(real)
    assert 2**30 <= m < 2**31
    assert m * 2.0 ** -(31 + s) == pytest.approx(real, rel=1e-8)


def test_quantize_multiplier_negative_and_zero():
    m, s = quantize_multiplier(-0.3)
    assert m < 0 and m * 2.0 ** -(31 + s) == pytest.approx(-0.3, rel=1e-8)
    assert quantize_multiplier(0.0) == (0, 0)


def test_multiply_by_quantized_multiplier_rounds_to_nearest():
    m, s = quantize_multiplier(0.37)
    x = np.arange(-1000, 1000, dtype=np.int64)
    exact = [(int(v) * m + (1 << (30 + s))) >> (31 + s) for v in x]
    np.testing.assert_array_equal(
        multiply_by_quantized_multiplier(x, m, s), exact
    )
    # ... which is the real product rounded, up to the 2**-31 multiplier error
    assert np.abs(np.array(exact) - x * 0.37).max() <= 0.5 + 1e-6


def test_multiply_by_quantized_multiplier_no_overflow_for_wide_accumulators():
    m, s = quantize_multiplier(0.9)
    x = np.array([2**60, -(2**60)], dtype=np.int64)
    got = multiply_by_quantized_multiplier(x, m, s)
    expected = [round(int(v) * m / 2 ** (31 + s)) for v in x]
    np.testing.assert_array_equal(got, expected)


def test_qparams_dict_roundtrip():
    qp = QParams(0.5, 3, 0, 255)
    assert QParams.from_dict(qp.to_dict()) == qp
