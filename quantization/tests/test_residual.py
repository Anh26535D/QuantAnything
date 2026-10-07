"""Residual streams: shared wide grid and the delayed add (fused epilogue)."""

import numpy as np
import onnxruntime as ort
import pytest

from quantization import (
    GemmQuantize,
    OnnxGraph,
    QuantContainer,
    float_ops,
    graph_passes,
    int_ops,
)
from quantization.quant_container import QuantContainer as QC
from quantization.tests.models import ModelBuilder, build_vit

D = 16


def _stream_model(blow_up=60.0, blocks=3, seed=0):
    """``x -> [x + Linear(x)] * blocks`` where block 2 makes one huge channel.

    The huge channel is the "massive activation" that wrecks a narrow
    residual grid for every other channel.
    """
    b = ModelBuilder(seed)
    x = "x"
    for k in range(blocks):
        w = (b.rng.standard_normal((D, D)) * 0.2).astype(np.float32)
        if k == 1:
            w[:, 3] *= blow_up
        y = b.op("MatMul", [x, b.const(w)])
        y = b.op("Add", [y, b.weight(D, scale=0.1)])
        x = b.op("Add", [x, y])
    return b.build("x", [2, 5, D], x, [2, 5, D])


def _data(seed, shape=(2, 5, D)):
    return (
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    )


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_residual_streams_are_found():
    c = QuantContainer(build_vit(batch=2, depth=3))
    groups = c.residual_groups()
    assert len(groups) == 1
    assert len(groups[0]) == 1 + 2 * 3  # embedding add + 2 per block


def test_fusion_moves_the_residual_add_into_the_linear_layer():
    model = _stream_model()
    plain = OnnxGraph.from_model(model)
    fused = OnnxGraph.from_model(model)
    n = graph_passes.fuse_residual_add(fused)
    assert n == 3
    ops = [x.op_type for x in fused.nodes]
    assert ops == ["LinearLayer"] * 3 and len(plain.nodes) == 6
    assert all(len(x.inputs) == 4 for x in fused.nodes)
    x = _data(0)
    np.testing.assert_allclose(
        float_ops.run_float(fused, x), float_ops.run_float(plain, x), atol=1e-5
    )


def test_fused_vit_matches_onnxruntime_in_float():
    model = build_vit(batch=2, depth=2, ln_style="op", gelu_style="op")
    x = (
        np.random.default_rng(1)
        .standard_normal((2, 3, 32, 32))
        .astype(np.float32)
    )
    ref = ort.InferenceSession(model.SerializeToString()).run(
        None, {"images": x}
    )[0]
    c = QuantContainer(model, residual="accumulate")
    assert c.fused_residuals == 4
    np.testing.assert_allclose(c.run_graph(x, "float"), ref, atol=1e-5)


def test_shared_grid_and_identity_multiplier_for_the_running_sum():
    c = QuantContainer(
        _stream_model(), "int8", residual="accumulate", residual_bits=16
    )
    c.calibrate([_data(1), _data(2)])
    (group,) = c.residual_groups()
    grids = {tuple(vars(l.out_qp[0]).values()) for l in group}
    assert len(grids) == 1  # one accumulator grid
    assert all(isinstance(l, GemmQuantize) and l.has_residual for l in group)
    # the first block takes its residual from the (int8) network input and
    # has to rescale it; every later add keeps the running sum untouched
    assert [l._same_grid for l in group] == [False, True, True]
    assert c.qparams[group[0].node.outputs[0]].qmax == 2**15 - 1


def _score_without_massive_channel(out, ref, channel=3):
    keep = [i for i in range(D) if i != channel]
    a, b = out[..., keep].reshape(-1), ref[..., keep].reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.parametrize("backend_native", [True, False])
def test_delayed_add_beats_a_narrow_residual_with_a_massive_channel(
    backend_native, monkeypatch
):
    from quantization import native

    if backend_native and not native.available():
        pytest.skip("native kernels missing")
    if not backend_native:
        monkeypatch.setattr(native, "_lib", None)
    model = _stream_model(blow_up=200.0)
    cal = [_data(s) for s in (1, 2, 3)]
    x = cal[0]
    ref = QuantContainer(model).run_graph(x, "float")

    def score(**kw):
        c = QuantContainer(model, "int8", **kw)
        c.calibrate(cal)
        return _score_without_massive_channel(c.run_graph(x, "integer"), ref)

    narrow = score()
    wide_add = score(precision={"Add": "int16"})
    delayed = score(residual="accumulate", residual_bits=16)
    # the huge channel stretches the grid; the other channels pay for it
    assert delayed > wide_add > narrow
    assert delayed > 0.9999


def test_integer_path_agrees_with_fake_quant_for_the_fused_layer():
    c = QuantContainer(
        _stream_model(), "int8", residual="accumulate", residual_bits=20
    )
    cal = [_data(s) for s in (1, 2)]
    c.calibrate(cal)
    fake, integer = (
        c.run_graph(cal[0], "quantize"),
        c.run_graph(cal[0], "integer"),
    )
    # one-step differences are amplified by the large weights of the stream
    assert np.abs(fake - integer).max() <= 5e-4 * np.abs(fake).max()


def test_lazy_mode_keeps_the_exact_sum_of_the_terms():
    model = _stream_model()
    cal = [_data(s) for s in (1, 2)]
    c = QuantContainer(model, "int8", residual="lazy")
    c.calibrate(cal)
    ref = c.run_graph(cal[0], "float")
    assert c.fused_residuals == 0  # nothing fused
    assert _cos(c.run_graph(cal[0], "integer"), ref) > 0.9


def test_precision_overrides_apply_by_op_type_and_by_layer_name():
    model = build_vit(batch=1, depth=1, ln_style="op", gelu_style="op")
    c = QuantContainer(
        model, "int8", precision={"LayerNormalization": "int16"}
    )
    name = next(l.name for l in c.layers if l.node.op_type == "Gemm")
    c2 = QuantContainer(model, "int8", precision={name: "int16"})
    rng = np.random.default_rng(0)
    cal = [rng.standard_normal((1, 3, 32, 32)).astype(np.float32)]
    for cont in (c, c2):
        cont.calibrate(cal)
    ln = next(l for l in c.layers if l.node.op_type == "LayerNormalization")
    assert ln.out_qp[0].qmax == 2**15 - 1
    gemm = next(l for l in c2.layers if l.name == name)
    assert gemm.out_qp[0].qmax == 2**15 - 1
    other = next(
        l for l in c2.layers if l.node.op_type == "LayerNormalization"
    )
    assert other.out_qp[0].qmax == 127


def test_wide_inputs_do_not_overflow_the_integer_layer_norm():
    rng = np.random.default_rng(3)
    c = 192
    scale = 2.0**-14  # 24-bit grid
    x = rng.normal(0, 400.0, (6, c)) + 30
    codes = np.round(x / scale).astype(np.int64)
    xr = codes * scale
    g, b = np.ones(c), np.zeros(c)
    ref = (xr - xr.mean(-1, keepdims=True)) / np.sqrt(
        xr.var(-1, keepdims=True) + 1e-6
    )
    out_scale = 1 / 1000
    mult, shift, post = int_ops.prepare_layer_norm(g, b, out_scale)
    got = (
        int_ops.layer_norm(
            codes, scale, 1e-6, mult, shift, post, 0, -(2**15), 2**15 - 1
        )
        * out_scale
    )
    assert np.abs(got - ref).max() < 0.05


def test_wide_input_gemm_uses_the_wide_accumulator():
    b = ModelBuilder(0)
    y = b.op("MatMul", ["x", b.weight(D, D)])
    y = b.op("Add", [y, b.weight(D, scale=0.1)])
    model = b.build("x", [2, 5, D], y, [2, 5, D])
    c = QuantContainer(model, "int8", precision={"nothing": "int8"})
    c.calibrate([_data(0)])
    layer = c.layers[0]
    assert not layer.wide  # int8 in, small K: int32
    # a 16-bit input grid forces the int64 path
    from quantization.quant_layers import qp_bits

    assert qp_bits(layer.in_qp[0]) == 8
    c16 = QuantContainer(model, "int16")
    c16.calibrate([_data(0)])
    assert c16.layers[0].wide


def test_exported_model_expands_the_fused_residual():
    model = _stream_model()
    g = OnnxGraph.from_model(model)
    graph_passes.fuse_residual_add(g)
    exported = g.to_model()
    x = _data(0)
    got = ort.InferenceSession(exported.SerializeToString()).run(
        None, {"x": x}
    )[0]
    ref = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[
        0
    ]
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_invalid_residual_mode_is_rejected():
    with pytest.raises(ValueError, match="residual"):
        QC(_stream_model(), residual="sometimes")


def test_every_vit_add_is_a_plain_integer_addition():
    """Same scale and zero point on both operands: no multiplier, no shift."""
    model = build_vit(batch=2, depth=3, ln_style="op", gelu_style="op")
    rng = np.random.default_rng(0)
    cal = [rng.standard_normal((2, 3, 32, 32)).astype(np.float32)]
    c = QuantContainer(model, "int8", strict=True, residual="accumulate")
    c.calibrate(cal)
    report = c.residual_report()
    assert len(report) == 1 + 2 * 3
    assert not any(needs for _, needs in report)
    # the embedding add (patch tokens + cls + position embedding) included
    (group,) = c.residual_groups()
    embed = next(l for l in group if type(l).__name__ == "AddQuantize")
    grid = embed.out_qp[0]
    assert all(
        (qp.scale, qp.zp) == (grid.scale, grid.zp) for qp in embed.in_qp
    )
    assert embed._mult[0] == embed._mult[1] == (2**30, -1)  # exactly 1.0
    # and the integer result still tracks the float model
    out = c.run_graph(cal[0], "integer")
    ref = c.run_graph(cal[0], "float")
    assert np.abs(out - ref).max() < 0.3


def test_graph_inputs_cannot_be_aligned_and_are_reported():
    model = _stream_model()
    c = QuantContainer(model, "int8", residual="accumulate")
    c.calibrate([_data(1), _data(2)])
    report = dict(c.residual_report())
    first = next(iter(report))
    assert report[first] is True and not any(
        v for k, v in report.items() if k != first
    )
