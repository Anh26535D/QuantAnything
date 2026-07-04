import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from quantization.quant_layers import (
    BatchNormalizationQuantize,
    DenseQuantize,
    ActivationQuantize,
    AddQuantize,
    ConcatenateQuantize,
    GlobalAveragePooling2DQuantize,
    ShapingQuantize
)

def test_bn_quantize():
    bn = keras.layers.BatchNormalization(axis=-1, name="test_bn")
    # Initialize
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
    bn(tf.convert_to_tensor(x_init))
    
    # Run calibration
    quant_bn = BatchNormalizationQuantize(bn, quant_type="int8")
    quant_bn.calibrate(x_init)
    quant_bn.finalize_calibration()
    
    # Check outputs
    x_infer = np.random.uniform(-1.0, 1.0, (1, 4, 4, 3)).astype(np.float32)
    out = quant_bn.quantized_infer(x_infer)
    assert out.shape == (1, 4, 4, 3)

def test_dense_quantize():
    dense = keras.layers.Dense(units=5, name="test_dense")
    x_init = np.random.uniform(-1.0, 1.0, (2, 10)).astype(np.float32)
    dense(tf.convert_to_tensor(x_init))
    
    quant_dense = DenseQuantize(dense, quant_type="int8", per_channel=True)
    quant_dense.calibrate(x_init)
    quant_dense.finalize_calibration()
    
    assert len(quant_dense.w_scale) == 5
    
    x_infer = np.random.uniform(-1.0, 1.0, (1, 10)).astype(np.float32)
    out = quant_dense.quantized_infer(x_infer)
    assert out.shape == (1, 5)

def test_activation_quantize():
    act = keras.layers.Activation("swish", name="test_act")
    
    quant_act = ActivationQuantize(act, quant_type="int8")
    x_init = np.random.uniform(-5.0, 5.0, (2, 5)).astype(np.float32)
    quant_act.calibrate(x_init)
    quant_act.finalize_calibration()
    
    out = quant_act.quantized_infer(x_init)
    assert out.shape == (2, 5)

def test_add_quantize():
    add = keras.layers.Add(name="test_add")
    x1 = np.random.uniform(-2.0, 2.0, (2, 4)).astype(np.float32)
    x2 = np.random.uniform(-3.0, 3.0, (2, 4)).astype(np.float32)
    
    quant_add = AddQuantize(add, quant_type="int8")
    quant_add.calibrate([x1, x2])
    quant_add.finalize_calibration()
    
    assert len(quant_add.in_scale_list) == 2
    
    out = quant_add.quantized_infer([x1, x2])
    assert out.shape == (2, 4)

def test_concatenate_quantize():
    concat = keras.layers.Concatenate(axis=-1, name="test_concat")
    x1 = np.random.uniform(-2.0, 2.0, (2, 4, 3)).astype(np.float32)
    x2 = np.random.uniform(-3.0, 3.0, (2, 4, 2)).astype(np.float32)
    
    quant_concat = ConcatenateQuantize(concat, quant_type="int8")
    quant_concat.calibrate([x1, x2])
    quant_concat.finalize_calibration()
    
    assert len(quant_concat.in_scale_list) == 2
    
    out = quant_concat.quantized_infer([x1, x2])
    assert out.shape == (2, 4, 5)

def test_gap_quantize():
    gap = keras.layers.GlobalAveragePooling2D(name="test_gap")
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4, 3)).astype(np.float32)
    gap(tf.convert_to_tensor(x_init))
    
    quant_gap = GlobalAveragePooling2DQuantize(gap, quant_type="int8")
    quant_gap.calibrate(x_init)
    quant_gap.finalize_calibration()
    
    out = quant_gap.quantized_infer(x_init)
    assert out.shape == (2, 3)

def test_shaping_quantize():
    reshape = keras.layers.Reshape((16,), name="test_reshape")
    x_init = np.random.uniform(-1.0, 1.0, (2, 4, 4)).astype(np.float32)
    reshape(tf.convert_to_tensor(x_init))
    
    quant_shape = ShapingQuantize(reshape, quant_type="int8")
    quant_shape.calibrate(x_init)
    quant_shape.finalize_calibration()
    
    # Scale and ZP must match exactly
    assert quant_shape.out_scale == quant_shape.in_scale
    assert quant_shape.out_zp == quant_shape.in_zp
    
    out = quant_shape.quantized_infer(x_init)
    assert out.shape == (2, 16)
