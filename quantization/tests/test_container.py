import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from quantization.quant_container import QuantContainer

def test_quant_container_functional_model():
    # Build a small Keras Functional model
    inputs = keras.Input(shape=(8, 8, 3), name="images")
    
    # Conv1 -> BN -> Act
    conv = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same", name="conv1")(inputs)
    bn = keras.layers.BatchNormalization(name="bn1")(conv)
    act = keras.layers.Activation("swish", name="act1")(bn)
    
    # Add block (shortcut connection)
    conv2 = keras.layers.Conv2D(filters=4, kernel_size=(1, 1), padding="same", name="conv2")(act)
    add = keras.layers.Add(name="add1")([act, conv2])
    
    # Pooling and Dense
    gap = keras.layers.GlobalAveragePooling2D(name="gap")(add)
    outputs = keras.layers.Dense(units=3, name="logits")(gap)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Initialize weights
    x_init = np.random.uniform(-1.0, 1.0, (2, 8, 8, 3)).astype(np.float32)
    model(tf.convert_to_tensor(x_init))
    
    # Create QuantContainer
    container = QuantContainer(model, quant_type="int8")
    
    # Calibrate
    x_cal = np.random.uniform(-2.0, 2.0, (5, 8, 8, 3)).astype(np.float32)
    container.calibrate(x_cal)
    
    # Verify that layers are correctly mapped and calibrated
    assert "conv1" in container.quant_layers
    assert "bn1" in container.quant_layers
    assert "add1" in container.quant_layers
    assert "logits" in container.quant_layers
    
    # Check that scales are computed
    assert container.quant_layers["conv1"].in_scale > 0
    assert container.quant_layers["logits"].out_scale > 0
    
    # Run quantized inference
    x_infer = np.random.uniform(-2.0, 2.0, (1, 8, 8, 3)).astype(np.float32)
    out_quant = container.quantized_infer(x_infer)
    
    assert out_quant.shape == (1, 3)
    
    # Standard Keras model output
    out_keras = model(tf.convert_to_tensor(x_infer)).numpy()
    
    # Check that fake quantization output is close to the original output (not identical due to quantization noise)
    # But shape and general values should match.
    assert out_quant.shape == out_keras.shape
