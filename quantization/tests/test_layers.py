"""Layer-level checks: integer path vs fake-quant path vs float reference."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from quantization import (
    ActivationQuantize,
    AddQuantize,
    BaseQuantLayer,
    BatchNormalizationQuantize,
    ConcatQuantize,
    ConvQuantize,
    GemmQuantize,
    GlobalAveragePoolQuantize,
    MulQuantize,
    QuantContainer,
    ShapingQuantize,
)
from quantization.tests.models import ModelBuilder

RNG = np.random.default_rng(0)


def _model(b, out, in_shape, extra_inputs=()):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)]
    inputs += [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
        for n, s in extra_inputs
    ]
    outs = out if isinstance(out, list) else [out]
    graph = helper.make_graph(
        b.nodes,
        "g",
        inputs,
        [
            helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
            for o in outs
        ],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    return model


def _calibrated(model, quant_type, calib, **kw):
    container = QuantContainer(model, quant_type, **kw)
    container.calibrate(calib)
    return container


def _out_qp(container):
    return container.qparams[container.graph.outputs[0]]


def _assert_int_matches_fake(container, x, lsb=1.0):
    """Integer result within ``lsb`` output steps of the fake-quant result."""
    fake = container.run_graph(x, "quantize")
    integer = container.run_graph(x, "integer")
    step = _out_qp(container).scale
    float_noise = 4 * np.finfo(np.float32).eps * np.abs(fake).max()
    assert np.abs(fake - integer).max() <= lsb * step * 1.001 + float_noise
    return fake, integer


QUANT_TYPES = ["int8", "uint8", "int16"]


def _x(shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, shape).astype(np.float32)


# ------------------------------------------------------------------- conv ----
CONVS = [
    dict(cin=3, cout=4, k=3, stride=1, groups=1),
    dict(cin=4, cout=6, k=3, stride=2, groups=1),
    dict(cin=6, cout=6, k=3, stride=1, groups=6),  # depthwise
    dict(cin=4, cout=8, k=1, stride=1, groups=2),
    dict(cin=3, cout=4, k=3, stride=1, groups=1, dilation=2, pad=2),
]


@pytest.mark.parametrize("backend_name", ["native", "numpy"])
@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize("cfg", CONVS)
def test_conv_integer_matches_fake_quant(
    cfg, quant_type, per_channel, backend_name, monkeypatch
):
    from quantization import native

    if backend_name == "numpy":
        monkeypatch.setattr(native, "_lib", None)
    elif not native.available():
        pytest.skip("native kernels missing")
    b = ModelBuilder(1)
    y = b.conv(
        "x",
        cfg["cin"],
        cfg["cout"],
        cfg["k"],
        cfg["stride"],
        cfg.get("pad"),
        cfg["groups"],
        dilation=cfg.get("dilation", 1),
    )
    shape = ["N", cfg["cin"], 10, 10]
    x = _x((2, cfg["cin"], 10, 10))
    # the test sample is part of the calibration set, so nothing saturates
    c = _calibrated(
        _model(b, y, shape),
        quant_type,
        [_x((4, cfg["cin"], 10, 10)), x],
        per_channel=per_channel,
    )
    assert isinstance(c.quant_layers[c.graph.nodes[0].name], ConvQuantize)
    fake, integer = _assert_int_matches_fake(c, x)
    # fake quantization stays close to the float model
    ref = c.run_graph(x, "float")
    assert np.abs(fake - ref).max() < 0.1 * np.abs(ref).max() + 1e-3


def test_conv_layer_exposes_quantization_parameters():
    b = ModelBuilder(1)
    y = b.conv("x", 3, 5, 3)
    c = _calibrated(_model(b, y, ["N", 3, 8, 8]), "int8", [_x((2, 3, 8, 8))])
    layer = c.layers[0]
    assert layer.in_zp == 0 and layer.out_zp == 0
    assert len(layer.w_scale) == 5 and np.all(layer.w_zp == 0)
    np.testing.assert_allclose(layer.b_scale, layer.in_scale * layer.w_scale)
    assert layer.w_centered.min() >= -128 and layer.w_centered.max() <= 127
    c2 = _calibrated(
        _model(b, y, ["N", 3, 8, 8]),
        "int8",
        [_x((2, 3, 8, 8))],
        per_channel=False,
    )
    assert np.ndim(c2.layers[0].w_scale) == 0


def test_conv_without_bias_and_16bit_bias_range():
    b = ModelBuilder(1)
    y = b.conv("x", 3, 4, 3, bias=False)
    c = _calibrated(_model(b, y, ["N", 3, 8, 8]), "int16", [_x((2, 3, 8, 8))])
    layer = c.layers[0]
    assert layer.bias_int is None and layer.b_qmax == 2**63 - 1
    _assert_int_matches_fake(c, _x((1, 3, 8, 8)))


def test_16bit_conv_does_not_overflow_the_accumulator():
    # K = 3*3*64 with 16-bit operands needs the int64 path.
    b = ModelBuilder(1)
    y = b.conv("x", 64, 8, 3)
    c = _calibrated(
        _model(b, y, ["N", 64, 6, 6]), "int16", [_x((2, 64, 6, 6))]
    )
    assert c.layers[0].wide
    x = _x((1, 64, 6, 6))
    c.calibrate([x])
    fake, integer = _assert_int_matches_fake(c, x)
    assert np.abs(integer - c.run_graph(x, "float")).max() < 1e-2


# ------------------------------------------------------------------- gemm ----
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
def test_gemm_integer_matches_fake_quant(quant_type):
    b = ModelBuilder(2)
    y = b.op(
        "Gemm",
        ["x", b.weight(7, 12), b.weight(7, scale=0.2)],
        transB=1,
        alpha=0.5,
        beta=2.0,
    )
    c = _calibrated(_model(b, y, ["N", 12]), quant_type, [_x((8, 12))])
    assert isinstance(c.layers[0], GemmQuantize)
    assert len(c.layers[0].w_scale) == 7
    _assert_int_matches_fake(c, _x((3, 12)))


def test_matmul_with_batch_dims():
    b = ModelBuilder(2)
    y = b.op("MatMul", ["x", b.weight(6, 5)])
    c = _calibrated(_model(b, y, ["N", 4, 6]), "int8", [_x((3, 4, 6))])
    assert isinstance(c.layers[0], GemmQuantize)
    fake, integer = _assert_int_matches_fake(c, _x((2, 4, 6)))
    assert integer.shape == (2, 4, 5)


# ----------------------------------------------------------- batch norm ----
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
def test_batchnorm_with_negative_gamma(quant_type):
    b = ModelBuilder(3)
    c_ = 4
    y = b.op(
        "BatchNormalization",
        [
            "x",
            b.const(np.array([1.5, -0.8, 0.3, -2.0], np.float32)),
            b.weight(c_),
            b.weight(c_),
            b.const(np.array([0.5, 1.0, 2.0, 0.7], np.float32)),
        ],
        epsilon=1e-3,
    )
    c = _calibrated(
        _model(b, y, ["N", c_, 5, 5]), quant_type, [_x((3, c_, 5, 5))]
    )
    assert isinstance(c.layers[0], BatchNormalizationQuantize)
    assert (c.layers[0].q_mult < 0).any()
    fake, integer = _assert_int_matches_fake(c, _x((2, c_, 5, 5)))
    ref = c.run_graph(_x((2, c_, 5, 5)), "float")
    assert ref.shape == integer.shape


# ----------------------------------------------------------- activations ----
@pytest.mark.parametrize("quant_type", ["int8", "uint8", "int16"])
@pytest.mark.parametrize("op", ["Sigmoid", "Relu", "Tanh", "LeakyRelu"])
def test_activation_lut_is_exact(op, quant_type):
    b = ModelBuilder(0)
    y = b.op(op, ["x"])
    c = _calibrated(
        _model(b, y, ["N", 3, 6]), quant_type, [_x((4, 3, 6), -3, 3)]
    )
    assert isinstance(c.layers[0], ActivationQuantize)
    fake, integer = _assert_int_matches_fake(c, _x((2, 3, 6), -3, 3), lsb=0)
    np.testing.assert_array_equal(fake, integer)


def test_clip_activation():
    b = ModelBuilder(0)
    y = b.op("Clip", ["x", b.const(np.float32(0.0)), b.const(np.float32(1.5))])
    c = _calibrated(_model(b, y, ["N", 8]), "int8", [_x((6, 8), -3, 3)])
    assert isinstance(c.layers[0], ActivationQuantize)
    out = c.run_graph(_x((4, 8), -3, 3), "integer")
    assert out.min() >= 0 and out.max() <= 1.5 + 1e-6


def test_large_bit_width_falls_back_to_generic_layer():
    b = ModelBuilder(0)
    y = b.op("Sigmoid", ["x"])
    c = _calibrated(_model(b, y, ["N", 8]), "int20", [_x((6, 8))])
    assert type(c.layers[0]) is BaseQuantLayer
    c.run_graph(_x((2, 8)), "integer")


# -------------------------------------------------------- element-wise ----
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize("op", ["Add", "Sub", "Mul"])
def test_binary_ops_between_two_branches(op, quant_type):
    b = ModelBuilder(4)
    p = b.conv("x", 3, 4, 3)
    q = b.op("Relu", [b.conv("x", 3, 4, 1)])
    y = b.op(op, [p, q])
    c = _calibrated(
        _model(b, y, ["N", 3, 6, 6]), quant_type, [_x((3, 3, 6, 6))]
    )
    kind = MulQuantize if op == "Mul" else AddQuantize
    assert isinstance(c.quant_layers[c.graph.nodes[-1].name], kind)
    _assert_int_matches_fake(c, _x((2, 3, 6, 6)), lsb=2)


def test_constant_operand_and_broadcasting():
    b = ModelBuilder(4)
    scale = b.const(
        np.array([1.0, 2.0, 0.5, -1.0], np.float32).reshape(1, 4, 1, 1)
    )
    y = b.op("Add", [b.op("Mul", ["x", scale]), b.const(np.float32(0.25))])
    x = _x((2, 4, 5, 5))
    c = _calibrated(
        _model(b, y, ["N", 4, 5, 5]), "int8", [_x((3, 4, 5, 5)), x]
    )
    fake, integer = _assert_int_matches_fake(c, x, lsb=2)
    ref = c.run_graph(x, "float")
    assert np.abs(integer - ref).max() < 0.1 * np.abs(ref).max()


@pytest.mark.parametrize("quant_type", ["int8", "uint8"])
def test_concat_rescales_inputs_to_one_grid(quant_type):
    b = ModelBuilder(5)
    p = b.conv("x", 3, 2, 3)  # small range
    q = b.op("Mul", [b.conv("x", 3, 3, 3), b.const(np.float32(10.0))])
    y = b.op("Concat", [p, q], axis=1)
    c = _calibrated(
        _model(b, y, ["N", 3, 6, 6]), quant_type, [_x((3, 3, 6, 6))]
    )
    assert isinstance(c.quant_layers[c.graph.nodes[-1].name], ConcatQuantize)
    fake, integer = _assert_int_matches_fake(c, _x((2, 3, 6, 6)), lsb=2)
    assert integer.shape == (2, 5, 6, 6)


def test_global_average_pool():
    b = ModelBuilder(6)
    y = b.op("GlobalAveragePool", ["x"])
    x = _x((2, 5, 7, 7))
    c = _calibrated(
        _model(b, y, ["N", 5, 7, 7]), "int8", [_x((3, 5, 7, 7)), x]
    )
    assert isinstance(c.layers[0], GlobalAveragePoolQuantize)
    fake, integer = _assert_int_matches_fake(c, x)
    assert integer.shape == (2, 5, 1, 1)
    assert np.abs(integer - x.mean(axis=(2, 3), keepdims=True)).max() < 0.05


# ---------------------------------------------------------------- shaping ----
@pytest.mark.parametrize("quant_type", ["int8", "uint8"])
def test_pad_uses_zero_point_as_padding_value(quant_type):
    b = ModelBuilder(0)
    pads = b.const(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64))
    y = b.op("Pad", ["x", pads])
    c = _calibrated(
        _model(b, y, ["N", 2, 4, 4]), quant_type, [_x((3, 2, 4, 4), 0.1, 3.0)]
    )
    assert isinstance(c.layers[0], ShapingQuantize)
    x = _x((1, 2, 4, 4), 0.1, 3.0)
    integer = c.run_graph(x, "integer")
    assert integer.shape == (1, 2, 6, 6)
    border = np.concatenate(
        [integer[0, :, 0].ravel(), integer[0, :, -1].ravel()]
    )
    np.testing.assert_array_equal(border, 0.0)  # real zero stays real zero
    _assert_int_matches_fake(c, x)


def test_maxpool_is_scale_preserving():
    b = ModelBuilder(0)
    y = b.op(
        "MaxPool",
        ["x"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    c = _calibrated(_model(b, y, ["N", 2, 8, 8]), "int8", [_x((3, 2, 8, 8))])
    assert isinstance(c.layers[0], ShapingQuantize)
    assert c.layers[0].out_scale == c.layers[0].in_scale
    _assert_int_matches_fake(c, _x((2, 2, 8, 8)), lsb=0)


def test_split_outputs_share_input_parameters():
    b = ModelBuilder(0)
    a, d = b.op("Split", ["x"], n_out=2, axis=1)
    y = b.op("Add", [a, b.op("Relu", [d])])
    c = _calibrated(_model(b, y, ["N", 4, 3, 3]), "int8", [_x((3, 4, 3, 3))])
    split = c.layers[0]
    assert isinstance(split, ShapingQuantize)
    assert [qp for qp in split.out_qp] == [split.in_qp[0]] * 2
    _assert_int_matches_fake(c, _x((2, 4, 3, 3)), lsb=2)


def test_softmax_uses_dequantize_float_requantize_fallback():
    b = ModelBuilder(0)
    y = b.op("Softmax", ["x"], axis=1)
    c = _calibrated(_model(b, y, ["N", 6]), "int8", [_x((8, 6))])
    assert type(c.layers[0]) is BaseQuantLayer
    x = _x((3, 6))
    integer = c.run_graph(x, "integer")
    np.testing.assert_allclose(integer.sum(axis=1), 1.0, atol=0.1)
    _assert_int_matches_fake(c, x)
