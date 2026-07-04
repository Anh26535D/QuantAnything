import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from quantization.quant_container import QuantContainer
from quantization.utils import quantize, dequantize

def test_integer_only_inference():
    # Create a small model with Conv2D, BatchNormalization, Activation, Add, Concatenate, GlobalAveragePooling2D, Dense
    inputs = keras.Input(shape=(3, 32, 32), name="images")
    x = keras.layers.Conv2D(16, (3, 3), padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.Activation("silu")(x)
    
    x2 = keras.layers.Conv2D(16, (3, 3), padding="same")(x1)
    # Add skip connection
    x_add = keras.layers.Add()([x1, x2])
    
    x_concat = keras.layers.Concatenate(axis=1)([x1, x_add])
    
    x_pool = keras.layers.GlobalAveragePooling2D()(x_concat)
    outputs = keras.layers.Dense(10)(x_pool)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Initialize container with int8
    container = QuantContainer(model, quant_type="int8")
    
    # Create calibration data
    calib_data = [np.random.uniform(0.0, 1.0, (4, 3, 32, 32)).astype(np.float32) for _ in range(2)]
    container.calibrate(calib_data)
    
    # Run test sample
    test_sample = np.random.uniform(0.0, 1.0, (1, 3, 32, 32)).astype(np.float32)
    
    # Run fake-quantization inference (mode="quantize")
    out_fake = container.quantized_infer(test_sample, mode="quantize")
    
    # Run integer-only inference (mode="integer")
    out_integer = container.quantized_infer(test_sample, mode="integer").astype(np.float32)
    
    # Verify shape and type
    assert out_fake.shape == out_integer.shape
    assert out_fake.dtype == out_integer.dtype
    
    # Verify outputs are close (max difference should be small because both simulate the same quantization parameters)
    diff = np.max(np.abs(out_fake - out_integer))
    print(f"Fake vs Integer Max Difference: {diff}")
    assert diff < 0.5
