import pytest
import numpy as np
from quantization.utils import parse_quant_type, calculate_scale_zp, fake_quantize
from quantization.quant_layers import InputQuantize

def test_parse_quant_type():
    # Test bitwidth and signedness parsing
    assert parse_quant_type("int8") == (8, True)
    assert parse_quant_type("uint8") == (8, False)
    assert parse_quant_type("int9") == (9, True)
    assert parse_quant_type("uint12") == (12, False)
    
    with pytest.raises(ValueError):
        parse_quant_type("float32")

def test_calculate_scale_zp_int8():
    # Test symmetric signed int8
    scale, zp, qmin, qmax = calculate_scale_zp(-10.0, 10.0, "int8")
    assert zp == 0
    assert qmin == -128
    assert qmax == 127
    assert abs(scale - (10.0 / 127.0)) < 1e-6

def test_calculate_scale_zp_uint8():
    # Test asymmetric unsigned uint8
    scale, zp, qmin, qmax = calculate_scale_zp(-2.0, 8.0, "uint8")
    assert qmin == 0
    assert qmax == 255
    assert abs(scale - (10.0 / 255.0)) < 1e-6
    assert zp == int(np.round(2.0 / scale))

def test_calculate_scale_zp_int9():
    # Test symmetric signed int9
    scale, zp, qmin, qmax = calculate_scale_zp(-100.0, 50.0, "int9")
    assert zp == 0
    assert qmin == -256
    assert qmax == 255
    assert abs(scale - (100.0 / 255.0)) < 1e-6

def test_input_quantize_int8():
    # Create InputQuantize layer with default int8
    layer = InputQuantize(original_layer=None, quant_type="int8")
    
    # Calibrate with range [-5.0, 5.0]
    x_cal = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    layer.calibrate(x_cal)
    layer.finalize_calibration()
    
    assert layer.in_scale > 0
    assert layer.in_zp == 0
    
    # Run inference
    x_infer = np.array([-3.0, 1.2, 4.5], dtype=np.float32)
    x_quant = layer.quantized_infer(x_infer)
    
    # Quantized values should match fake_quantize
    expected = fake_quantize(x_infer, layer.in_scale, layer.in_zp, layer.in_qmin, layer.in_qmax)
    np.testing.assert_allclose(x_quant, expected)

def test_input_quantize_uint8():
    # Create InputQuantize layer with uint8
    layer = InputQuantize(original_layer=None, quant_type="uint8")
    
    # Calibrate
    x_cal = np.array([-2.0, 0.0, 8.0], dtype=np.float32)
    layer.calibrate(x_cal)
    layer.finalize_calibration()
    
    assert layer.in_scale > 0
    assert layer.in_zp > 0
    
    x_infer = np.array([-1.0, 4.0, 7.5], dtype=np.float32)
    x_quant = layer.quantized_infer(x_infer)
    expected = fake_quantize(x_infer, layer.in_scale, layer.in_zp, layer.in_qmin, layer.in_qmax)
    np.testing.assert_allclose(x_quant, expected)
