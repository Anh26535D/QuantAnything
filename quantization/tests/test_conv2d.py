import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from quantization.quant_layers import Conv2DQuantize

def test_conv2d_quantize_int8_per_channel():
    # Define a simple Conv2D layer
    # Input shape: (batch_size, height, width, channels) = (2, 4, 4, 3)
    # Output channels: 2
    conv = keras.layers.Conv2D(
        filters=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="test_conv"
    )
    
    # Initialize weights
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
    conv(tf.convert_to_tensor(x_init))
    
    # Set explicit weights
    W = np.random.uniform(-10.0, 10.0, (3, 3, 3, 2)).astype(np.float32)
    bias = np.array([2.5, -3.5], dtype=np.float32)
    conv.set_weights([W, bias])
    
    # Create quantized Conv2D wrapper (default signed int8, per_channel=True)
    quant_layer = Conv2DQuantize(conv, quant_type="int8", per_channel=True)
    
    # Calibrate
    x_cal = np.random.uniform(-5.0, 5.0, (5, 4, 4, 3)).astype(np.float32)
    quant_layer.calibrate(x_cal)
    quant_layer.finalize_calibration()
    
    # Assertions
    assert quant_layer.in_scale > 0
    assert quant_layer.in_zp == 0
    assert quant_layer.out_scale > 0
    assert quant_layer.out_zp == 0
    
    assert len(quant_layer.w_scale) == 2
    assert np.all(quant_layer.w_zp == 0)
    
    # Bias scale should be input_scale * weight_scale
    assert np.allclose(quant_layer.b_scale, quant_layer.in_scale * quant_layer.w_scale)
    
    # Inference
    x_infer = np.random.uniform(-5.0, 5.0, (1, 4, 4, 3)).astype(np.float32)
    out_quant = quant_layer.quantized_infer(x_infer)
    
    assert out_quant.shape == (1, 2, 2, 2)
    # Outputs should be bounded
    assert np.min(out_quant) >= quant_layer.out_scale * quant_layer.out_qmin
    assert np.max(out_quant) <= quant_layer.out_scale * quant_layer.out_qmax

def test_conv2d_quantize_int8_per_tensor():
    conv = keras.layers.Conv2D(
        filters=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="test_conv_tensor"
    )
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
    conv(tf.convert_to_tensor(x_init))
    
    # Create quantized Conv2D wrapper (per_channel=False)
    quant_layer = Conv2DQuantize(conv, quant_type="int8", per_channel=False)
    
    # Calibrate
    x_cal = np.random.uniform(-5.0, 5.0, (5, 4, 4, 3)).astype(np.float32)
    quant_layer.calibrate(x_cal)
    quant_layer.finalize_calibration()
    
    assert isinstance(quant_layer.w_scale, float) or quant_layer.w_scale.ndim == 0
    assert quant_layer.w_zp == 0
    
    x_infer = np.random.uniform(-5.0, 5.0, (1, 4, 4, 3)).astype(np.float32)
    out_quant = quant_layer.quantized_infer(x_infer)
    assert out_quant.shape == (1, 2, 2, 2)

def test_conv2d_quantize_uint8():
    conv = keras.layers.Conv2D(
        filters=2,
        kernel_size=(3, 3),
        use_bias=False,
        name="test_conv_uint8"
    )
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
    conv(tf.convert_to_tensor(x_init))
    
    # Create quantized wrapper (uint8, per_channel=True)
    quant_layer = Conv2DQuantize(conv, quant_type="uint8", per_channel=True)
    
    # Use a range crossing zero to ensure zp is strictly positive
    x_cal = np.random.uniform(-2.0, 8.0, (5, 4, 4, 3)).astype(np.float32)
    quant_layer.calibrate(x_cal)
    quant_layer.finalize_calibration()
    
    assert quant_layer.in_zp > 0
    assert quant_layer.out_zp > 0
    
    x_infer = np.random.uniform(0.0, 10.0, (1, 4, 4, 3)).astype(np.float32)
    out_quant = quant_layer.quantized_infer(x_infer)
    assert out_quant.shape == (1, 2, 2, 2)
