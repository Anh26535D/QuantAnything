"""Piecewise-linear fits and the integer-only softmax / LayerNorm."""

import numpy as np
import pytest

from quantization import QuantContainer, int_ops, native, pwl
from quantization.float_ops import _erf
from quantization.tests.models import ModelBuilder, build_vit, build_yolo_like

FUNCS = {
    "gelu": lambda x: 0.5 * x * (1 + _erf(x / np.sqrt(2))),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
}


def test_fit_uses_at_most_four_segments_and_rejects_more():
    p = pwl.fit_pwl(np.tanh, -4, 4)
    assert p.segments == 4 and len(p.x) == 5
    assert np.all(np.diff(p.x) > 0)
    with pytest.raises(ValueError):
        pwl.fit_pwl(np.tanh, -4, 4, segments=5)
    assert pwl.fit_pwl(np.tanh, -4, 4, segments=2).segments == 2


@pytest.mark.parametrize(
    "name,bound",
    [("gelu", 0.05), ("silu", 0.03), ("sigmoid", 0.03), ("tanh", 0.09)],
)
def test_activation_fit_error_bounds(name, bound):
    p = pwl.fit_pwl(FUNCS[name], -6, 6)
    assert pwl.error_report(FUNCS[name], p, -6, 6)[0] < bound


def test_piecewise_linear_functions_are_fitted_exactly():
    relu = pwl.fit_pwl(lambda x: np.maximum(x, 0), -3, 3)
    assert pwl.error_report(lambda x: np.maximum(x, 0), relu, -3, 3)[0] < 1e-9
    hs = lambda x: np.clip(0.2 * x + 0.5, 0, 1)  # noqa: E731  (3 segments)
    assert pwl.error_report(hs, pwl.fit_pwl(hs, -5, 5), -5, 5)[0] < 1e-9


@pytest.mark.parametrize(
    "fn,lo,hi,rel",
    [
        (lambda u: 2.0**-u, 0, 1, 0.003),  # exp via 2^-f
        (lambda m: 1 / m, 1, 2, 0.006),  # softmax reciprocal
        (lambda m: 1 / np.sqrt(m), 1, 4, 0.008),  # layer-norm rsqrt
    ],
)
def test_relative_fits_for_exp_reciprocal_and_rsqrt(fn, lo, hi, rel):
    p = pwl.fit_pwl(fn, lo, hi, relative=True)
    assert pwl.error_report(fn, p, lo, hi)[1] < rel


def test_fit_is_better_than_uniform_knots():
    f = FUNCS["gelu"]
    best = pwl.fit_pwl(f, -4, 4)
    uniform = pwl.PWL(np.linspace(-4, 4, 5), f(np.linspace(-4, 4, 5)))
    assert (
        pwl.error_report(f, best, -4, 4)[0]
        < pwl.error_report(f, uniform, -4, 4)[0]
    )


def test_degenerate_range_gives_a_usable_function():
    p = pwl.fit_pwl(np.tanh, 0.5, 0.5)
    assert np.isfinite(p(np.array([0.5]))).all()


def test_integer_pwl_matches_the_float_fit_and_numpy_fallback(monkeypatch):
    f = FUNCS["silu"]
    p = pwl.fit_pwl(f, -5, 5)
    ip = pwl.to_integer(p, 1 / 512, 0, 1 / 512, 0, -32768, 32767)
    assert ip.segments <= 4
    x = np.arange(-2560, 2560, dtype=np.int32)
    got = ip(x) / 512
    assert np.abs(got - p(x / 512)).max() < 2 / 512 + 1e-6
    if native.available():
        fast = ip(x)
        monkeypatch.setattr(native, "_lib", None)
        np.testing.assert_array_equal(fast, ip(x))


def test_native_pwl_rejects_more_than_four_segments():
    z = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="at most 4"):
        native.pwl(np.zeros(3, np.int32), z, z, z, z, -8, 7)


# ------------------------------------------------- integer softmax / LN ----
@pytest.mark.parametrize(
    "bits,scale,sigma", [(8, 0.1, 3.0), (16, 0.002, 3.0), (16, 0.001, 6.0)]
)
def test_integer_softmax_matches_float(bits, scale, sigma):
    rng = np.random.default_rng(0)
    codes = np.round(rng.normal(0, sigma, (32, 197)) / scale).astype(np.int64)
    real = codes * scale
    ref = np.exp(real - real.max(-1, keepdims=True))
    ref /= ref.sum(-1, keepdims=True)
    out_scale = 1 / (2**bits - 1)
    got = int_ops.softmax(codes, scale, out_scale, 0, 0, 2**bits - 1)
    assert got.dtype == np.int32 and got.min() >= 0
    assert np.abs(got * out_scale - ref).max() < 0.006
    # the float emulation used by fake quantization has the same error
    emu = int_ops.softmax_float(real)
    assert np.abs(emu - ref).max() < 0.006
    assert np.abs(emu - got * out_scale).max() < 0.002 + out_scale


def test_integer_softmax_is_shift_invariant_and_handles_extremes():
    codes = np.array([[5, 5, 5, 5], [0, 0, 0, -(10**9)], [10**8, 0, 0, 0]])
    p = int_ops.softmax(codes, 1e-3, 1 / 255, 0, 0, 255) / 255
    np.testing.assert_allclose(p[0], 0.25, atol=0.005)
    np.testing.assert_allclose(p[1][:3], 1 / 3, atol=0.01)
    assert p[1][3] == 0 and p[2][0] > 0.99
    shifted = int_ops.softmax(codes + 1234, 1e-3, 1 / 255, 0, 0, 255) / 255
    np.testing.assert_array_equal(p, shifted)


@pytest.mark.parametrize("bits,scale", [(8, 0.05), (16, 0.0005)])
def test_integer_layer_norm_matches_float(bits, scale):
    rng = np.random.default_rng(1)
    c = 192
    x = rng.normal(0.3, 1.5, (50, c)) * np.linspace(0.3, 3, c)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    codes = np.clip(np.round(x / scale), lo, hi).astype(np.int64)
    xr = codes * scale
    g, b = 1 + 0.2 * rng.standard_normal(c), 0.1 * rng.standard_normal(c)
    ref = (xr - xr.mean(-1, keepdims=True)) / np.sqrt(
        xr.var(-1, keepdims=True) + 1e-6
    ) * g + b
    out_scale = np.abs(ref).max() / hi
    mult, shift, post = int_ops.prepare_layer_norm(g, b, out_scale)
    got = (
        int_ops.layer_norm(codes, scale, 1e-6, mult, shift, post, 0, lo, hi)
        * out_scale
    )
    assert np.sqrt(np.mean((got - ref) ** 2)) < 0.02 + out_scale
    emu = int_ops.layer_norm_float(xr, g, b, 1e-6)
    assert np.abs(emu - ref).max() < 0.05
    assert np.abs(emu - got).max() < 0.05 + 2 * out_scale


def test_integer_layer_norm_ignores_the_zero_point():
    rng = np.random.default_rng(2)
    codes = rng.integers(-100, 100, (4, 64)).astype(np.int64)
    g, b = np.ones(64), np.zeros(64)
    mult, shift, post = int_ops.prepare_layer_norm(g, b, 0.05)
    a = int_ops.layer_norm(codes, 0.05, 1e-6, mult, shift, post, 0, -128, 127)
    c = int_ops.layer_norm(
        codes + 20, 0.05, 1e-6, mult, shift, post, 0, -128, 127
    )
    assert np.abs(a - c).max() <= 1


# --------------------------------------------------------- full pipeline ----
def test_strict_mode_rejects_float_fallbacks():
    b = ModelBuilder(0)
    y = b.op("Softplus", [b.op("Tanh", ["x"])])  # fused: stays integer
    y = b.op("ReduceMean", [y], axes=[1], keepdims=1)  # no integer layer
    model = b.build("x", [2, 4, 3], y, [2, 1, 3])
    from quantization.quant_container import NotIntegerError

    with pytest.raises(NotIntegerError, match="ReduceMean"):
        QuantContainer(model, strict=True)
    report = QuantContainer(model).integer_report()
    assert [ok for *_, ok in report] == [True, False]


def test_vit_runs_fully_integer_and_is_bit_exact_across_backends(monkeypatch):
    model = build_vit(batch=2, depth=2, dim=32, heads=4)
    rng = np.random.default_rng(0)
    cal = [
        rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
        for _ in range(2)
    ]
    c = QuantContainer(model, "int16", strict=True)
    c.calibrate(cal)
    fast = c.run_graph(cal[0], "integer")
    if native.available():
        monkeypatch.setattr(native, "_lib", None)
        np.testing.assert_array_equal(fast, c.run_graph(cal[0], "integer"))


def test_cnn_activations_become_four_segment_integer_functions():
    model = build_yolo_like(size=32)
    c = QuantContainer(model, "int16", strict=False)
    rng = np.random.default_rng(0)
    c.calibrate([rng.uniform(0, 1, (2, 3, 32, 32)).astype(np.float32)])
    acts = [l for l in c.layers if type(l).__name__ == "ActivationQuantize"]
    assert acts and all(l.pwl_int.segments <= 4 for l in acts)
    # worst case over the fit domain / RMS error on the calibration data
    assert all(l.pwl_error < 0.12 for l in acts)
    assert all(l.pwl_rmse < 0.03 for l in acts)
    assert [l.node.op_type for l in c.float_fallbacks()] == []


def test_save_and_load_keep_the_pwl_ranges(tmp_path):
    model = build_yolo_like(size=32)
    rng = np.random.default_rng(0)
    cal = [rng.uniform(0, 1, (2, 3, 32, 32)).astype(np.float32)]
    c = QuantContainer(model, "int16")
    c.calibrate(cal)
    c.save(tmp_path / "q.json")
    loaded = QuantContainer.load(model, tmp_path / "q.json")
    assert loaded.nonlinear == "pwl"
    np.testing.assert_array_equal(
        c.run_graph(cal[0], "integer"), loaded.run_graph(cal[0], "integer")
    )
