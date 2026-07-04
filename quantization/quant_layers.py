import numpy as np
from quantization.utils import (
    calculate_scale_zp,
    fake_quantize,
    parse_quant_type,
    quantize,
    dequantize,
    quantize_multiplier,
    multiply_by_quantized_multiplier
)

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

    def finalize_calibration(self, container=None):
        """
        Calculates scale and zero points after calibration.
        Inherits input scale/zp from the preceding layer if container context is provided.
        """
        if container is not None and not isinstance(self, InputQuantize):
            inputs = self.original_layer.input
            if not isinstance(inputs, list):
                tensor = inputs
                if hasattr(tensor, "_keras_history"):
                    producer = tensor._keras_history[0]
                    if producer.name in container.quant_layers:
                        prod_q = container.quant_layers[producer.name]
                        if prod_q.out_scale is not None:
                            self.in_scale = prod_q.out_scale
                            self.in_zp = prod_q.out_zp
                            self.in_qmin = prod_q.out_qmin
                            self.in_qmax = prod_q.out_qmax
                            
        if self.in_scale is None:
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

    def quantized_infer_integer(self, inputs):
        """
        Fallback implementation for integer-only inference.
        """
        if self.in_scale is None:
            raise ValueError(f"Layer {self.name} is not calibrated.")
        import tensorflow as tf
        
        if isinstance(inputs, (list, tuple)):
            if hasattr(self, 'in_scale_list') and self.in_scale_list:
                x = [
                    dequantize(val, self.in_scale_list[i], self.in_zp_list[i])
                    for i, val in enumerate(inputs)
                ]
            else:
                x = [
                    dequantize(val, self.in_scale, self.in_zp)
                    for val in inputs
                ]
            inputs_tf = [tf.convert_to_tensor(v, dtype=tf.float32) for v in x]
        else:
            x = dequantize(inputs, self.in_scale, self.in_zp)
            inputs_tf = tf.convert_to_tensor(x, dtype=tf.float32)
            
        outputs_tf = self.original_layer(inputs_tf)
        outputs_np = outputs_tf.numpy()
        return quantize(outputs_np, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)


class InputQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def calibrate(self, inputs):
        self.in_min, self.in_max = self.update_min_max(inputs, self.in_min, self.in_max)
        self.out_min, self.out_max = self.in_min, self.in_max
        return inputs

    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        # Input layer outputs same as inputs, so copy scale/zp
        self.out_scale = self.in_scale
        self.out_zp = self.in_zp
        self.out_qmin = self.in_qmin
        self.out_qmax = self.in_qmax

    def quantized_infer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"InputQuantize layer {self.name} is not calibrated.")
        return fake_quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"InputQuantize layer {self.name} is not calibrated.")
        return quantize(inputs, self.in_scale, self.in_zp, self.in_qmin, self.in_qmax)


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

    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        
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
            if num_bits > 8:
                self.b_qmin = -2**63
                self.b_qmax = 2**63 - 1
            else:
                self.b_qmin = -2**31
                self.b_qmax = 2**31 - 1
            self.b_zp = 0
            if self.per_channel:
                self.b_scale = self.in_scale * self.w_scale
            else:
                self.b_scale = self.in_scale * self.w_scale

        # Precompute integer multipliers and shifts
        out_channels = W.shape[-1]
        if self.per_channel:
            self.q_mult = np.zeros(out_channels, dtype=np.int64)
            self.shift = np.zeros(out_channels, dtype=np.int64)
            for c in range(out_channels):
                real_mult = (self.in_scale * self.w_scale[c]) / self.out_scale
                self.q_mult[c], self.shift[c] = quantize_multiplier(real_mult)
        else:
            real_mult = (self.in_scale * self.w_scale) / self.out_scale
            self.q_mult, self.shift = quantize_multiplier(real_mult)

        # Precompute bias_int for integer inference
        if has_bias:
            if self.per_channel:
                self.bias_int = np.round(bias / (self.in_scale * self.w_scale)).astype(np.int64)
            else:
                self.bias_int = np.round(bias / (self.in_scale * self.w_scale)).astype(np.int64)
        else:
            self.bias_int = None

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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"Conv2DQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        orig_weights = self.original_layer.get_weights()
        W = orig_weights[0]
        has_bias = len(orig_weights) > 1
        bias = orig_weights[1] if has_bias else None
        
        out_channels = W.shape[-1]
        
        # Quantize weights and bias
        q_W = np.zeros_like(W)
        if self.per_channel:
            for c in range(out_channels):
                q_W[..., c] = quantize(W[..., c], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax)
        else:
            q_W = quantize(W, self.w_scale, self.w_zp, self.w_qmin, self.w_qmax)
            
        q_bias = None
        if has_bias:
            q_bias = quantize(bias, self.b_scale, self.b_zp, self.b_qmin, self.b_qmax)
            
        # Run integer convolution
        in_centered = inputs - self.in_zp
        
        q_W_centered = np.zeros_like(q_W)
        if self.per_channel:
            for c in range(out_channels):
                q_W_centered[..., c] = q_W[..., c] - self.w_zp[c]
        else:
            q_W_centered = q_W - self.w_zp
            
        inputs_tf = tf.convert_to_tensor(in_centered, dtype=tf.float32)
        
        # Use quantized weights with zero bias; the bias will be added in integer arithmetic after scaling
        if has_bias:
            dummy_bias = np.zeros_like(bias)
            self.original_layer.set_weights([q_W_centered, dummy_bias])
        else:
            self.original_layer.set_weights([q_W_centered])
        
        outputs_tf = self.original_layer(inputs_tf)
        accum = np.round(outputs_tf.numpy()).astype(np.int64)
        
        self.original_layer.set_weights(orig_weights)
        
        # Scale back to output scale using integer multiply and shift (layout NCHW or NHWC)
        q_out = np.zeros_like(accum)
        data_format = getattr(self.original_layer, "data_format", "channels_last")
        channel_axis = 1 if data_format == "channels_first" else -1
        
        if self.per_channel:
            for c in range(out_channels):
                if channel_axis == 1:
                    acc_c = accum[:, c, ...]
                    # Add bias in the accumulator scale (in_scale * w_scale)
                    if self.bias_int is not None:
                        acc_c = acc_c + self.bias_int[c]
                    res = multiply_by_quantized_multiplier(acc_c, self.q_mult[c], self.shift[c])
                    q_out[:, c, ...] = res + self.out_zp
                else:
                    acc_c = accum[..., c]
                    if self.bias_int is not None:
                        acc_c = acc_c + self.bias_int[c]
                    res = multiply_by_quantized_multiplier(acc_c, self.q_mult[c], self.shift[c])
                    q_out[..., c] = res + self.out_zp
        else:
            if self.bias_int is not None:
                accum = accum + self.bias_int
            q_out = multiply_by_quantized_multiplier(accum, self.q_mult, self.shift) + self.out_zp
            
        return np.clip(q_out, self.out_qmin, self.out_qmax)


class BatchNormalizationQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        weights = self.original_layer.get_weights()
        gamma = weights[0]
        beta = weights[1]
        mean = weights[2]
        var = weights[3]
        epsilon = self.original_layer.epsilon
        
        self.A = gamma / np.sqrt(var + epsilon)
        self.B = beta - self.A * mean
        
        channels = self.A.shape[0]
        self.q_mult = np.zeros(channels, dtype=np.int64)
        self.shift = np.zeros(channels, dtype=np.int64)
        self.q_B = np.zeros(channels, dtype=np.int64)
        
        for c in range(channels):
            real_mult = (self.in_scale * self.A[c]) / self.out_scale
            self.q_mult[c], self.shift[c] = quantize_multiplier(real_mult)
            self.q_B[c] = int(np.round(self.B[c] / self.out_scale))

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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"BatchNormalizationQuantize layer {self.name} is not calibrated.")
            
        axis = self.original_layer.axis
        channel_axis = axis[0] if isinstance(axis, (list, tuple)) else axis
        if channel_axis < 0:
            channel_axis = inputs.ndim + channel_axis
            
        channels = inputs.shape[channel_axis]
        q_out = np.zeros_like(inputs)
        
        for c in range(channels):
            if channel_axis == 1 or channel_axis == inputs.ndim - 3:
                # NCHW slicing
                in_c = inputs[:, c, ...]
                in_centered = in_c - self.in_zp
                res = multiply_by_quantized_multiplier(in_centered, self.q_mult[c], self.shift[c])
                q_out[:, c, ...] = res + self.q_B[c] + self.out_zp
            else:
                # NHWC slicing
                in_c = inputs[..., c]
                in_centered = in_c - self.in_zp
                res = multiply_by_quantized_multiplier(in_centered, self.q_mult[c], self.shift[c])
                q_out[..., c] = res + self.q_B[c] + self.out_zp
            
        return np.clip(q_out, self.out_qmin, self.out_qmax)


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

    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        
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
            if num_bits > 8:
                self.b_qmin = -2**63
                self.b_qmax = 2**63 - 1
            else:
                self.b_qmin = -2**31
                self.b_qmax = 2**31 - 1
            self.b_zp = 0
            if self.per_channel:
                self.b_scale = self.in_scale * self.w_scale
            else:
                self.b_scale = self.in_scale * self.w_scale

        # Precompute integer multipliers and shifts
        out_features = W.shape[-1]
        if self.per_channel:
            self.q_mult = np.zeros(out_features, dtype=np.int64)
            self.shift = np.zeros(out_features, dtype=np.int64)
            for c in range(out_features):
                real_mult = (self.in_scale * self.w_scale[c]) / self.out_scale
                self.q_mult[c], self.shift[c] = quantize_multiplier(real_mult)
        else:
            real_mult = (self.in_scale * self.w_scale) / self.out_scale
            self.q_mult, self.shift = quantize_multiplier(real_mult)

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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"DenseQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        orig_weights = self.original_layer.get_weights()
        W = orig_weights[0]
        has_bias = len(orig_weights) > 1
        bias = orig_weights[1] if has_bias else None
        
        out_features = W.shape[-1]
        
        # Quantize weights and bias
        q_W = np.zeros_like(W)
        if self.per_channel:
            for c in range(out_features):
                q_W[:, c] = quantize(W[:, c], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax)
        else:
            q_W = quantize(W, self.w_scale, self.w_zp, self.w_qmin, self.w_qmax)
            
        q_bias = None
        if has_bias:
            q_bias = quantize(bias, self.b_scale, self.b_zp, self.b_qmin, self.b_qmax)
            
        # Run integer matrix multiplication
        in_centered = inputs - self.in_zp
        
        q_W_centered = np.zeros_like(q_W)
        if self.per_channel:
            for c in range(out_features):
                q_W_centered[:, c] = q_W[:, c] - self.w_zp[c]
        else:
            q_W_centered = q_W - self.w_zp
            
        inputs_tf = tf.convert_to_tensor(in_centered, dtype=tf.float32)
        
        weights_to_set = [q_W_centered, q_bias] if has_bias else [q_W_centered]
        self.original_layer.set_weights(weights_to_set)
        
        outputs_tf = self.original_layer(inputs_tf)
        accum = np.round(outputs_tf.numpy()).astype(np.int64)
        
        self.original_layer.set_weights(orig_weights)
        
        # Scale back to output scale using integer multiply and shift
        q_out = np.zeros_like(accum)
        if self.per_channel:
            for c in range(out_features):
                acc_c = accum[..., c]
                q_out[..., c] = multiply_by_quantized_multiplier(acc_c, self.q_mult[c], self.shift[c]) + self.out_zp
        else:
            q_out = multiply_by_quantized_multiplier(accum, self.q_mult, self.shift) + self.out_zp
            
        return np.clip(q_out, self.out_qmin, self.out_qmax)


class ActivationQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        # Precompute Lookup Table (LUT) for activation function
        num_bits, _ = parse_quant_type(self.quant_type)
        size = 2**num_bits
        qmin = -(2**(num_bits - 1))
        
        # Populate LUT
        q_in_arr = np.arange(qmin, qmin + size, dtype=np.int64)
        x_arr = (q_in_arr - self.in_zp) * self.in_scale
        
        import tensorflow as tf
        x_tf = tf.convert_to_tensor(x_arr, dtype=tf.float32)
        y_tf = self.original_layer(x_tf)
        y_arr = y_tf.numpy()
        
        self.lut = quantize(y_arr, self.out_scale, self.out_zp, self.out_qmin, self.out_qmax)

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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"ActivationQuantize layer {self.name} is not calibrated.")
            
        # Shift inputs to start at index 0
        num_bits, _ = parse_quant_type(self.quant_type)
        qmin = -(2**(num_bits - 1))
        
        idx = (inputs - qmin).astype(np.intp)
        idx = np.clip(idx, 0, len(self.lut) - 1)
        return self.lut[idx]


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

    def finalize_calibration(self, container=None):
        inherited = False
        if container is not None:
            inputs = self.original_layer.input
            scale_list = []
            zp_list = []
            qmin_list = []
            qmax_list = []
            for tensor in inputs:
                if hasattr(tensor, "_keras_history"):
                    producer = tensor._keras_history[0]
                    if producer.name in container.quant_layers:
                        prod_q = container.quant_layers[producer.name]
                        if prod_q.out_scale is not None:
                            scale_list.append(prod_q.out_scale)
                            zp_list.append(prod_q.out_zp)
                            qmin_list.append(prod_q.out_qmin)
                            qmax_list.append(prod_q.out_qmax)
            if len(scale_list) == len(inputs):
                self.in_scale_list = scale_list
                self.in_zp_list = zp_list
                self.in_qmin_list = qmin_list
                self.in_qmax = qmax_list  # Wait: in_qmax_list? Yes, let's keep name matching: in_qmax_list
                self.in_qmax_list = qmax_list
                inherited = True
                
        if not inherited:
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

        # Precompute input rescaling multipliers to output scale
        self.q_mult1, self.shift1 = quantize_multiplier(self.in_scale_list[0] / self.out_scale)
        self.q_mult2, self.shift2 = quantize_multiplier(self.in_scale_list[1] / self.out_scale)

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

    def quantized_infer_integer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"AddQuantize layer {self.name} is not calibrated.")
        
        q_in1, q_in2 = inputs[0], inputs[1]
        
        # Rescale both integer inputs to the output scale
        in1_centered = q_in1 - self.in_zp_list[0]
        in2_centered = q_in2 - self.in_zp_list[1]
        
        res1 = multiply_by_quantized_multiplier(in1_centered, self.q_mult1, self.shift1)
        res2 = multiply_by_quantized_multiplier(in2_centered, self.q_mult2, self.shift2)
        
        q_out = res1 + res2 + self.out_zp
        return np.clip(q_out, self.out_qmin, self.out_qmax)


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

    def finalize_calibration(self, container=None):
        # Attempt to inherit input scales from preceding layers if container is provided
        inherited = False
        if container is not None:
            inputs = self.original_layer.input
            if not isinstance(inputs, list):
                inputs = [inputs]
            scale_list = []
            zp_list = []
            qmin_list = []
            qmax_list = []
            for tensor in inputs:
                if hasattr(tensor, "_keras_history"):
                    producer = tensor._keras_history[0]
                    if producer.name in container.quant_layers:
                        prod_q = container.quant_layers[producer.name]
                        if prod_q.out_scale is not None:
                            scale_list.append(prod_q.out_scale)
                            zp_list.append(prod_q.out_zp)
                            qmin_list.append(prod_q.out_qmin)
                            qmax_list.append(prod_q.out_qmax)
            if len(scale_list) == len(inputs):
                self.in_scale_list = scale_list
                self.in_zp_list = zp_list
                self.in_qmin_list = qmin_list
                self.in_qmax_list = qmax_list
                inherited = True
        if not inherited:
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

        # Precompute input rescaling multipliers to output scale
        self.q_mult_list = []
        self.shift_list = []
        for i in range(len(self.in_scale_list)):
            qm, sh = quantize_multiplier(self.in_scale_list[i] / self.out_scale)
            self.q_mult_list.append(qm)
            self.shift_list.append(sh)

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

    def quantized_infer_integer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"ConcatenateQuantize layer {self.name} is not calibrated.")
            
        rescaled_inputs = []
        for i, q_in in enumerate(inputs):
            in_centered = q_in - self.in_zp_list[i]
            res = multiply_by_quantized_multiplier(in_centered, self.q_mult_list[i], self.shift_list[i]) + self.out_zp
            res_clipped = np.clip(res, self.out_qmin, self.out_qmax)
            rescaled_inputs.append(res_clipped)
            
        # Concatenate along the dynamic axis
        axis = getattr(self.original_layer, "axis", 1)
        return np.concatenate(rescaled_inputs, axis=axis)


class GlobalAveragePooling2DQuantize(BaseQuantLayer):
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        # Precompute rescaling from input to output scale
        self.q_mult, self.shift = quantize_multiplier(self.in_scale / self.out_scale)

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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"GlobalAveragePooling2DQuantize layer {self.name} is not calibrated.")
            
        data_format = getattr(self.original_layer, "data_format", "channels_last")
        pool_axes = (2, 3) if data_format == "channels_first" else (1, 2)
        
        H, W = inputs.shape[pool_axes[0]], inputs.shape[pool_axes[1]]
        in_centered = inputs - self.in_zp
        sum_x = np.sum(in_centered, axis=pool_axes)
        
        area = H * W
        # Average in the input scale
        avg_x = np.round(sum_x / area).astype(np.int64)
        
        # Rescale from input scale to output scale
        rescaled = multiply_by_quantized_multiplier(avg_x, self.q_mult, self.shift) + self.out_zp
        return np.clip(rescaled, self.out_qmin, self.out_qmax)


class ShapingQuantize(BaseQuantLayer):
    """
    Handles shape-only operations (Reshape, Flatten, ZeroPadding2D, Cropping2D, Lambda splits/slices).
    For these layers, the output scale/zero-point is mathematically identical to the input scale/zero-point.
    """
    def __init__(self, original_layer, quant_type="int8"):
        super().__init__(original_layer, quant_type)
        
    def finalize_calibration(self, container=None):
        if container is not None:
            # Try to inherit scale/zp from producer layer
            inputs = self.original_layer.input
            if isinstance(inputs, list):
                tensor = inputs[0]
            else:
                tensor = inputs
            if hasattr(tensor, "_keras_history"):
                producer = tensor._keras_history[0]
                if producer.name in container.quant_layers:
                    prod_q = container.quant_layers[producer.name]
                    print(f"[DEBUG INHERIT] Layer {self.name}: producer = {producer.name}, prod_q.out_scale = {prod_q.out_scale}")
                    if prod_q.out_scale is not None:
                        self.in_scale = prod_q.out_scale
                        self.in_zp = prod_q.out_zp
                        self.in_qmin = prod_q.out_qmin
                        self.in_qmax = prod_q.out_qmax
                        
        if self.in_scale is None:
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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None:
            raise ValueError(f"ShapingQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        # Shaping operations can receive single integer array or list of integer arrays (e.g. from Split)
        if isinstance(inputs, (list, tuple)):
            inputs_tf = [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs]
        else:
            inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
            
        outputs_tf = self.original_layer(inputs_tf)
        
        if isinstance(outputs_tf, list):
            return [tf.cast(tf.round(o), tf.int64).numpy() for o in outputs_tf]
        else:
            return tf.cast(tf.round(outputs_tf), tf.int64).numpy()


class DepthwiseConv2DQuantize(BaseQuantLayer):
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

    def finalize_calibration(self, container=None):
        super().finalize_calibration(container)
        
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
            # DepthwiseConv2D weight shape: (filter_height, filter_width, in_channels, depth_multiplier)
            in_channels = W.shape[-2]
            self.w_scale = np.zeros(in_channels)
            self.w_zp = np.zeros(in_channels, dtype=int)
            for c in range(in_channels):
                W_c = W[..., c, :]
                sc, zp, _, _ = calculate_scale_zp(W_c.min(), W_c.max(), self.quant_type)
                self.w_scale[c] = sc
                self.w_zp[c] = zp
        else:
            sc, zp, _, _ = calculate_scale_zp(W.min(), W.max(), self.quant_type)
            self.w_scale = sc
            self.w_zp = zp
            
        if has_bias:
            if num_bits > 8:
                self.b_qmin = -2**63
                self.b_qmax = 2**63 - 1
            else:
                self.b_qmin = -2**31
                self.b_qmax = 2**31 - 1
            self.b_zp = 0
            if self.per_channel:
                self.b_scale = self.in_scale * self.w_scale
            else:
                self.b_scale = self.in_scale * self.w_scale

        # Precompute integer multipliers and shifts
        in_channels = W.shape[-2]
        if self.per_channel:
            self.q_mult = np.zeros(in_channels, dtype=np.int64)
            self.shift = np.zeros(in_channels, dtype=np.int64)
            for c in range(in_channels):
                real_mult = (self.in_scale * self.w_scale[c]) / self.out_scale
                self.q_mult[c], self.shift[c] = quantize_multiplier(real_mult)
        else:
            real_mult = (self.in_scale * self.w_scale) / self.out_scale
            self.q_mult, self.shift = quantize_multiplier(real_mult)

    def quantized_infer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"DepthwiseConv2DQuantize layer {self.name} is not calibrated.")
            
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
            in_channels = W.shape[-2]
            for c in range(in_channels):
                fake_W[..., c, :] = fake_quantize(
                    W[..., c, :], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax
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

    def quantized_infer_integer(self, inputs):
        if self.in_scale is None or self.w_scale is None:
            raise ValueError(f"DepthwiseConv2DQuantize layer {self.name} is not calibrated.")
            
        import tensorflow as tf
        
        orig_weights = self.original_layer.get_weights()
        W = orig_weights[0]
        has_bias = len(orig_weights) > 1
        bias = orig_weights[1] if has_bias else None
        
        in_channels = W.shape[-2]
        
        # Quantize weights and bias
        q_W = np.zeros_like(W)
        if self.per_channel:
            for c in range(in_channels):
                q_W[..., c, :] = quantize(W[..., c, :], self.w_scale[c], self.w_zp[c], self.w_qmin, self.w_qmax)
        else:
            q_W = quantize(W, self.w_scale, self.w_zp, self.w_qmin, self.w_qmax)
            
        q_bias = None
        if has_bias:
            q_bias = quantize(bias, self.b_scale, self.b_zp, self.b_qmin, self.b_qmax)
            
        # Run integer depthwise convolution
        in_centered = inputs - self.in_zp
        
        q_W_centered = np.zeros_like(q_W)
        if self.per_channel:
            for c in range(in_channels):
                q_W_centered[..., c, :] = q_W[..., c, :] - self.w_zp[c]
        else:
            q_W_centered = q_W - self.w_zp
            
        inputs_tf = tf.convert_to_tensor(in_centered, dtype=tf.float32)
        
        weights_to_set = [q_W_centered, q_bias] if has_bias else [q_W_centered]
        self.original_layer.set_weights(weights_to_set)
        
        outputs_tf = self.original_layer(inputs_tf)
        accum = np.round(outputs_tf.numpy()).astype(np.int64)
        
        self.original_layer.set_weights(orig_weights)
        
        # Scale back to output scale using integer multiply and shift (layout NCHW or NHWC)
        q_out = np.zeros_like(accum)
        data_format = getattr(self.original_layer, "data_format", "channels_last")
        channel_axis = 1 if data_format == "channels_first" else -1
        
        if self.per_channel:
            for c in range(in_channels):
                if channel_axis == 1:
                    acc_c = accum[:, c, ...]
                    res = multiply_by_quantized_multiplier(acc_c, self.q_mult[c], self.shift[c]) + self.out_zp
                    q_out[:, c, ...] = res
                else:
                    acc_c = accum[..., c]
                    res = multiply_by_quantized_multiplier(acc_c, self.q_mult[c], self.shift[c]) + self.out_zp
                    q_out[..., c] = res
        else:
            q_out = multiply_by_quantized_multiplier(accum, self.q_mult, self.shift) + self.out_zp
            
        return np.clip(q_out, self.out_qmin, self.out_qmax)


class MultiplyQuantize(BaseQuantLayer):
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

    def finalize_calibration(self, container=None):
        # Attempt to inherit input scales from preceding layers if container is provided
        inherited = False
        if container is not None:
            inputs = self.original_layer.input
            if not isinstance(inputs, list):
                inputs = [inputs]
            scale_list = []
            zp_list = []
            qmin_list = []
            qmax_list = []
            for tensor in inputs:
                if hasattr(tensor, "_keras_history"):
                    producer = tensor._keras_history[0]
                    if producer.name in container.quant_layers:
                        prod_q = container.quant_layers[producer.name]
                        if prod_q.out_scale is not None:
                            scale_list.append(prod_q.out_scale)
                            zp_list.append(prod_q.out_zp)
                            qmin_list.append(prod_q.out_qmin)
                            qmax_list.append(prod_q.out_qmax)
            if len(scale_list) == len(inputs):
                self.in_scale_list = scale_list
                self.in_zp_list = zp_list
                self.in_qmin_list = qmin_list
                self.in_qmax_list = qmax_list
                inherited = True
        if not inherited:
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
        
        # Quantization multiplier: M = (S_x1 * S_x2) / S_y
        real_mult = (self.in_scale_list[0] * self.in_scale_list[1]) / self.out_scale
        self.q_mult, self.shift = quantize_multiplier(real_mult)

    def quantized_infer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"MultiplyQuantize layer {self.name} is not calibrated.")
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

    def quantized_infer_integer(self, inputs):
        if not self.in_scale_list:
            raise ValueError(f"MultiplyQuantize layer {self.name} is not calibrated.")
            
        q_in1, q_in2 = inputs[0], inputs[1]
        
        # Element-wise integer multiplication: (q1 - Z1) * (q2 - Z2)
        in1_centered = q_in1 - self.in_zp_list[0]
        in2_centered = q_in2 - self.in_zp_list[1]
        
        accum = in1_centered * in2_centered
        
        # Scale back to output scale using integer multiply and shift
        q_out = multiply_by_quantized_multiplier(accum, self.q_mult, self.shift) + self.out_zp
        return np.clip(q_out, self.out_qmin, self.out_qmax)
