import numpy as np
from quantization.utils import calculate_scale_zp, fake_quantize, parse_quant_type

class BaseQuantLayer:
    def __init__(self, original_layer, quant_type="int8"):
        self.original_layer = original_layer
        self.name = original_layer.name if original_layer else "layer"
        self.quant_type = quant_type
        
        # Min/max collected during calibration
        self.in_min = None
        self.in_max = None
        self.out_min = None
        self.out_max = None
        
        # Scale and zero point for inputs/outputs
        self.in_scale = None
        self.in_zp = None
        self.in_qmin = None
        self.in_qmax = None
        
        self.out_scale = None
        self.out_zp = None
        self.out_qmin = None
        self.out_qmax = None

    def update_min_max(self, x, current_min, current_max):
        if x is None:
            return current_min, current_max
        x_min = np.min(x)
        x_max = np.max(x)
        if current_min is None:
            new_min = x_min
            new_max = x_max
        else:
            new_min = min(current_min, x_min)
            new_max = max(current_max, x_max)
        return new_min, new_max

    def calibrate(self, inputs):
        """
        Runs float forward pass on input, collecting min/max stats.
        Returns the output of the original layer as numpy array.
        """
        # Collect input min/max
        if isinstance(inputs, (list, tuple)):
            # Subclasses with multiple inputs should handle this
            pass
        else:
            self.in_min, self.in_max = self.update_min_max(inputs, self.in_min, self.in_max)
        
        # Run original layer
        # Wrap execution in TensorFlow to handle Keras tensors if needed
        import tensorflow as tf
        
        # If inputs is a NumPy array, convert it to a TF tensor for the Keras layer
        if isinstance(inputs, np.ndarray):
            inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
        elif isinstance(inputs, (list, tuple)):
            inputs_tf = [tf.convert_to_tensor(i, dtype=tf.float32) if isinstance(i, np.ndarray) else i for i in inputs]
        else:
            inputs_tf = inputs
            
        outputs = self.original_layer(inputs_tf)
        
        # Convert output back to numpy array
        if hasattr(outputs, "numpy"):
            outputs_np = outputs.numpy()
        else:
            outputs_np = np.array(outputs)
            
        self.out_min, self.out_max = self.update_min_max(outputs_np, self.out_min, self.out_max)
        return outputs_np

    def finalize_calibration(self):
        """
        Calculates scale and zero points after calibration.
        """
        if self.in_min is not None:
            self.in_scale, self.in_zp, self.in_qmin, self.in_qmax = calculate_scale_zp(
                self.in_min, self.in_max, self.quant_type
            )
        if self.out_min is not None:
            self.out_scale, self.out_zp, self.out_qmin, self.out_qmax = calculate_scale_zp(
                self.out_min, self.out_max, self.quant_type
            )

    def quantized_infer(self, inputs):
        """
        Performs quantized forward inference.
        """
        if self.in_scale is None:
            raise ValueError(f"Layer {self.name} is not calibrated.")
        import tensorflow as tf
        
        if isinstance(inputs, (list, tuple)):
            if hasattr(self, 'in_scale_list') and self.in_scale_list:
                fake_in = [
                    fake_quantize(x, self.in_scale_list[i], self.in_zp_list[i], self.in_qmin_list[i], self.in_qmax_list[i])
                    for i, x in enumerate(inputs)
                ]
            else:
                fake_in = [
                    fake_quantize(x, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
                    for x in inputs
                ]
            inputs_tf = [tf.convert_to_tensor(v, dtype=tf.float32) for v in fake_in]
        else:
            fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
            inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
            
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class InputQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def calibrate(self, inputs):
        self.in_min, self.in_max = self.update_min_max(inputs, self.in_min, self.in_max)
        self.out_min, self.out_max = self.in_min, self.in_max
        return inputs

    def finalize_calibration(self):
        super().finalize_calibration()
        # Input layer outputs same as inputs, so copy scale/zp
        self.out_scale = self.in_scale
        self.out_zp = self.in_zp
        self.out_qmin = self.in_qmin
        self.out_qmax = self.in_qmax

    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"InputQuantize layer {self.name} is not calibrated.")
        return fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)


class Conv2DQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8", per_channel=True):
        super().__init__(original_layer, quant_type)
        self.per_channel = per_channel
        self.w_scale = None
        self.w_zp = None
        self.w_qmin = None
        self.w_qmax = None
        self.b_scale = None
        self.b_zp = None
        self.b_qmin = None
        self.b_qmax = None

    def finalize_calibration(self):
        super().finalize_calibration()
        
        weights = self.original_layer.get_weights()
        W = weights[0]
        has_bias = len(weights) > 1
        bias = weights[1] if has_bias else None
        
        num_bits, signed = parse_quant_type(self.quant_type)
        if signed:
            w_qmin = -(2**(num_bits - 1))
            w_qmax = 2**(num_bits - 1) - 1
        else:
            w_qmin = 0
            w_qmax = 2**num_bits - 1
            
        self.w_qmin, self.w_qmax = w_qmin, w_qmax
        
        if self.per_channel:
            # Conv2D kernel shape: (filter_height, filter_width, in_channels, out_channels)
            out_channels = W.shape[-1]
            self.w_scale = np.zeros(out_channels)
            self.w_zp = np.zeros(out_channels, dtype=int)
            for c in range(out_channels):
                W_c = W[..., c]
                sc, zp, _, _ = calculate_scale_zp(W_c.min(), W_c.max(), self.quant_type)
                self.w_scale[c] = sc
                self.w_zp[c] = zp
        else:
            sc, zp, _, _ = calculate_scale_zp(W.min(), W.max(), self.quant_type)
            self.w_scale = sc
            self.w_zp = zp
            
        if has_bias:
            self.b_qmin = -2**31
            self.b_qmax = 2**31 - 1
            self.b_zp = 0
            if self.per_channel:
                self.b_scale = self.in_scale * self.w_scale
            else:
                self.b_scale = self.in_scale * self.w_scale

    def quantized_infer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"Conv2DQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        orig_weights = self.original_layer.get_weights()
        W = orig_weights[0]
        has_bias = len(orig_weights) > 1
        bias = orig_weights[1] if has_bias else None
        
        # 1. Fake-quantize input
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        
        # 2. Fake-quantize weights
        fake_W = np.zeros_like(W)
        if self.per_channel:
            out_channels = W.shape[-1]
            for c in range(out_channels):
                fake_W[..., c] = fake_quantize(
                    W[..., c], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax
                )
        else:
            fake_W = fake_quantize(W, self.w_scale, self.w_zp, self.w_qmin, self.w_qmax)
            
        # 3. Fake-quantize bias
        fake_bias = None
        if has_bias:
            fake_bias = fake_quantize(bias, self.b_scale, self.b_zp, self.b_qmin, self.b_qmax)
            
        # 4. Run convolution with fake weights
        weights_to_set = [fake_W, fake_bias] if has_bias else [fake_W]
        self.original_layer.set_weights(weights_to_set)
        
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        self.original_layer.set_weights(orig_weights)
        
        # 5. Fake-quantize outputs
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class BatchNormalizationQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"BatchNormalizationQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        # 1. Fake-quantize input
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        
        # 2. Run original BN layer
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        # 3. Fake-quantize output
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class DenseQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8", per_channel=True):
        super().__init__(original_layer, quant_type)
        self.per_channel = per_channel
        self.w_scale = None
        self.w_zp = None
        self.w_qmin = None
        self.w_qmax = None
        self.b_scale = None
        self.b_zp = None
        self.b_qmin = None
        self.b_qmax = None

    def finalize_calibration(self):
        super().finalize_calibration()
        
        weights = self.original_layer.get_weights()
        W = weights[0]
        has_bias = len(weights) > 1
        bias = weights[1] if has_bias else None
        
        num_bits, signed = parse_quant_type(self.quant_type)
        if signed:
            w_qmin = -(2**(num_bits - 1))
            w_qmax = 2**(num_bits - 1) - 1
        else:
            w_qmin = 0
            w_qmax = 2**num_bits - 1
            
        self.w_qmin, self.w_qmax = w_qmin, w_qmax
        
        if self.per_channel:
            # Dense weight shape: (in_features, out_features)
            out_features = W.shape[-1]
            self.w_scale = np.zeros(out_features)
            self.w_zp = np.zeros(out_features, dtype=int)
            for c in range(out_features):
                W_c = W[:, c]
                sc, zp, _, _ = calculate_scale_zp(W_c.min(), W_c.max(), self.quant_type)
                self.w_scale[c] = sc
                self.w_zp[c] = zp
        else:
            sc, zp, _, _ = calculate_scale_zp(W.min(), W.max(), self.quant_type)
            self.w_scale = sc
            self.w_zp = zp
            
        if has_bias:
            self.b_qmin = -2**31
            self.b_qmax = 2**31 - 1
            self.b_zp = 0
            if self.per_channel:
                self.b_scale = self.in_scale * self.w_scale
            else:
                self.b_scale = self.in_scale * self.w_scale

    def quantized_infer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"DenseQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        orig_weights = self.original_layer.get_weights()
        W = orig_weights[0]
        has_bias = len(orig_weights) > 1
        bias = orig_weights[1] if has_bias else None
        
        # 1. Fake-quantize input
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        
        # 2. Fake-quantize weights
        fake_W = np.zeros_like(W)
        if self.per_channel:
            out_features = W.shape[-1]
            for c in range(out_features):
                fake_W[:, c] = fake_quantize(
                    W[:, c], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax
                )
        else:
            fake_W = fake_quantize(W, self.w_scale, self.w_zp, self.w_qmin, self.w_qmax)
            
        # 3. Fake-quantize bias
        fake_bias = None
        if has_bias:
            fake_bias = fake_quantize(bias, self.b_scale, self.b_zp, self.b_qmin, self.b_qmax)
            
        # 4. Swap weights, run dense layer, restore weights
        weights_to_set = [fake_W, fake_bias] if has_bias else [fake_W]
        self.original_layer.set_weights(weights_to_set)
        
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        self.original_layer.set_weights(orig_weights)
        
        # 5. Fake-quantize outputs
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class ActivationQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"ActivationQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        # 1. Fake-quantize input
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        
        # 2. Run activation
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        # 3. Fake-quantize output
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class AddQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        self.in_min_list = []
        self.in_max_list = []
        self.in_scale_list = []
        self.in_zp_list = []
        self.in_qmin_list = []
        self.in_qmax_list = []

    def calibrate(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            
        if not self.in_min_list:
            self.in_min_list = [None] * len(inputs)
            self.in_max_list = [None] * len(inputs)
            
        for i, inp in enumerate(inputs):
            self.in_min_list[i], self.in_max_list[i] = self.update_min_max(
                inp, self.in_min_list[i], self.in_max_list[i]
            )
            
        import tensorflow as tf
        inputs_tf = [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs]
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        self.out_min, self.out_max = self.update_min_max(outputs_np, self.out_min, self.out_max)
        return outputs_np

    def finalize_calibration(self):
        self.in_scale_list = []
        self.in_zp_list = []
        self.in_qmin_list = []
        self.in_qmax_list = []
        for i in range(len(self.in_min_list)):
            sc, zp, qmin, qmax = calculate_scale_zp(
                self.in_min_list[i], self.in_max_list[i], self.quant_type
            )
            self.in_scale_list.append(sc)
            self.in_zp_list.append(zp)
            self.in_qmin_list.append(qmin)
            self.in_qmax_list.append(qmax)
            
        if self.out_min is not None:
            self.out_scale, self.out_zp, self.out_qmin, self.out_qmax = calculate_scale_zp(
                self.out_min, self.out_max, self.quant_type
            )

    def quantized_infer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"AddQuantize layer {self.name} is not calibrated.")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            
        import tensorflow as tf
        
        fake_inputs = []
        for i, inp in enumerate(inputs):
            fake_inp = fake_quantize(
                inp, self.in_scale_list[i], self.in_zp_list[i], self.in_qmin_list[i], self.in_qmax_list[i]
            )
            fake_inputs.append(fake_inp)
            
        inputs_tf = [tf.convert_to_tensor(x, dtype=tf.float32) for x in fake_inputs]
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class ConcatenateQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        self.in_min_list = []
        self.in_max_list = []
        self.in_scale_list = []
        self.in_zp_list = []
        self.in_qmin_list = []
        self.in_qmax_list = []

    def calibrate(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            
        if not self.in_min_list:
            self.in_min_list = [None] * len(inputs)
            self.in_max_list = [None] * len(inputs)
            
        for i, inp in enumerate(inputs):
            self.in_min_list[i], self.in_max_list[i] = self.update_min_max(
                inp, self.in_min_list[i], self.in_max_list[i]
            )
            
        import tensorflow as tf
        inputs_tf = [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs]
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        self.out_min, self.out_max = self.update_min_max(outputs_np, self.out_min, self.out_max)
        return outputs_np

    def finalize_calibration(self):
        self.in_scale_list = []
        self.in_zp_list = []
        self.in_qmin_list = []
        self.in_qmax_list = []
        for i in range(len(self.in_min_list)):
            sc, zp, qmin, qmax = calculate_scale_zp(
                self.in_min_list[i], self.in_max_list[i], self.quant_type
            )
            self.in_scale_list.append(sc)
            self.in_zp_list.append(zp)
            self.in_qmin_list.append(qmin)
            self.in_qmax_list.append(qmax)
            
        if self.out_min is not None:
            self.out_scale, self.out_zp, self.out_qmin, self.out_qmax = calculate_scale_zp(
                self.out_min, self.out_max, self.quant_type
            )

    def quantized_infer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"ConcatenateQuantize layer {self.name} is not calibrated.")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            
        import tensorflow as tf
        
        fake_inputs = []
        for i, inp in enumerate(inputs):
            fake_inp = fake_quantize(
                inp, self.in_scale_list[i], self.in_zp_list[i], self.in_qmin_list[i], self.in_qmax_list[i]
            )
            fake_inputs.append(fake_inp)
            
        inputs_tf = [tf.convert_to_tensor(x, dtype=tf.float32) for x in fake_inputs]
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class GlobalAveragePooling2DQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"GlobalAveragePooling2DQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        # 1. Fake-quantize input
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        
        # 2. Run pool operation
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        
        # 3. Fake-quantize output
        fake_out = fake_quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)
        return fake_out


class ShapingQuantize(BaseQuantLayer):
    """
    Handles shape-only operations (Reshape, Flatten, ZeroPadding2D, Cropping2D, Lambda splits/slices).
    For these layers, the output scale/zero-point is mathematically identical to the input scale/zero-point.
    """
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def finalize_calibration(self):
        super().finalize_calibration()
        # Output scale/zp copy input
        self.out_scale = self.in_scale
        self.out_zp = self.in_zp
        self.out_qmin = self.in_qmin
        self.out_qmax = self.in_qmax
        
    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"ShapingQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        fake_in = fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)
        inputs_tf = tf.convert_to_tensor(fake_in, dtype=tf.float32)
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        return outputs_np
