import numpy as np
import tensorflow as tf
from tensorflow import keras
from quantization.utils import dequantize
from quantization.quant_layers import (
    BaseQuantLayer,
    InputQuantize,
    Conv2DQuantize,
    BatchNormalizationQuantize,
    DenseQuantize,
    ActivationQuantize,
    AddQuantize,
    ConcatenateQuantize,
    GlobalAveragePooling2DQuantize,
    ShapingQuantize,
    DepthwiseConv2DQuantize,
    MultiplyQuantize
)

def map_layer_to_quant(layer, quant_type="int8"):
    """
    Maps a standard Keras layer to its corresponding quantized layer wrapper.
    """
    name = type(layer).__name__
    
    if isinstance(layer, keras.layers.InputLayer):
        return InputQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.DepthwiseConv2D):
        return DepthwiseConv2DQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Conv2D):
        return Conv2DQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.BatchNormalization):
        return BatchNormalizationQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Dense):
        return DenseQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Activation):
        return ActivationQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Add):
        return AddQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Multiply):
        return MultiplyQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.Concatenate):
        return ConcatenateQuantize(layer, quant_type)
    elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
        return GlobalAveragePooling2DQuantize(layer, quant_type)
    elif isinstance(layer, (keras.layers.Reshape, keras.layers.Flatten, keras.layers.ZeroPadding2D, keras.layers.Cropping2D)):
        return ShapingQuantize(layer, quant_type)
    elif "Lambda" in name:
        # Map split Lambda layers to ShapingQuantize (preserves scale/zp)
        # Map other Lambdas (like Softmax) to BaseQuantLayer (recalculates scale/zp)
        if "split" in layer.name.lower():
            return ShapingQuantize(layer, quant_type)
        else:
            return BaseQuantLayer(layer, quant_type)
    else:
        # Fallback based on input connectivity
        try:
            inputs = layer.input
            if isinstance(inputs, list) and len(inputs) > 1:
                return AddQuantize(layer, quant_type)
        except Exception:
            pass
        return BaseQuantLayer(layer, quant_type)


class QuantContainer:
    """
    Main container class representing our custom quantization computational graph.
    """
    def __init__(self, keras_model, quant_type="int8"):
        self.model = keras_model
        self.quant_type = quant_type
        self.quant_layers = {}
        
        # Instantiate quantized wrappers for all layers
        for layer in self.model.layers:
            self.quant_layers[layer.name] = map_layer_to_quant(layer, self.quant_type)

    def calibrate(self, x_calib):
        """
        Runs calibration batches to collect min/max activations and finalize scale/zp parameters.
        x_calib can be a numpy array or a list/generator of batches.
        """
        # Support single array or list of arrays
        if isinstance(x_calib, np.ndarray):
            batches = [x_calib]
        else:
            batches = x_calib
            
        print("Starting calibration forward passes...")
        for batch_idx, batch in enumerate(batches):
            self.run_graph(batch, mode="calibrate")
            
        print("Finalizing calibration scale and zero-point parameters...")
        for name, q_layer in self.quant_layers.items():
            q_layer.finalize_calibration(container=self)
        print("Calibration completed successfully.")

    def quantized_infer(self, x, mode="quantize"):
        """
        Runs quantized forward inference on the custom computational graph.
        mode can be "quantize" (fake quantization) or "integer" (integer-only arithmetic).
        """
        return self.run_graph(x, mode=mode)

    def run_graph(self, x, mode="quantize"):
        """
        Executes the topological graph pass layer by layer.
        """
        tensor_values = {}
        
        # Feed inputs
        # If x is a single input, map it to the model's first input tensor
        if not isinstance(x, dict):
            input_tensor = self.model.inputs[0]
            tensor_values[id(input_tensor)] = x
        else:
            for k, v in x.items():
                for inp in self.model.inputs:
                    if inp.name == k:
                        tensor_values[id(inp)] = v
                        break
                        
        # Iterate over layers in topological order
        for layer in self.model.layers:
            # InputLayer outputs are already fed into tensor_values
            if isinstance(layer, keras.layers.InputLayer):
                layer_output = layer.output
                q_layer = self.quant_layers[layer.name]
                if mode == "calibrate":
                    tensor_values[id(layer_output)] = q_layer.calibrate(x)
                elif mode == "integer":
                    tensor_values[id(layer_output)] = q_layer.quantized_infer_integer(x)
                elif mode == "quantize":
                    tensor_values[id(layer_output)] = q_layer.quantized_infer(x)
                else:
                    if id(layer_output) not in tensor_values or tensor_values[id(layer_output)] is x:
                        if not isinstance(x, dict):
                            tensor_values[id(layer_output)] = x
                continue
                
            q_layer = self.quant_layers[layer.name]
            
            # Retrieve inputs for this layer
            inputs = layer.input
            if isinstance(inputs, list):
                # Retrieve value for each input tensor in the list
                inp_vals = [tensor_values[id(t)] for t in inputs]
            else:
                inp_vals = tensor_values[id(inputs)]
                
            # Execute layer operation
            if mode == "calibrate":
                out_val = q_layer.calibrate(inp_vals)
            elif mode == "quantize":
                out_val = q_layer.quantized_infer(inp_vals)
            elif mode == "integer":
                out_val = q_layer.quantized_infer_integer(inp_vals)
            elif mode == "float":
                # Fallback to standard float forward
                inputs_tf = [tf.convert_to_tensor(v, dtype=tf.float32) for v in inp_vals] if isinstance(inp_vals, list) else tf.convert_to_tensor(inp_vals, dtype=tf.float32)
                outputs_tf = layer(inputs_tf)
                out_val = outputs_tf.numpy()
            else:
                raise ValueError(f"Unknown graph mode: {mode}")
                
            # Store outputs
            outputs = layer.output
            if isinstance(outputs, list):
                for j, out_t in enumerate(outputs):
                    tensor_values[id(out_t)] = out_val[j]
            else:
                tensor_values[id(outputs)] = out_val
                
        # Return outputs matching model structure
        if len(self.model.outputs) == 1:
            out_val = tensor_values[id(self.model.outputs[0])]
        else:
            out_val = [tensor_values[id(out)] for out in self.model.outputs]
            
        if mode == "integer":
            # Dequantize final integer outputs back to float for user/evaluation comparison
            final_layer = self.model.layers[-1]
            q_layer = self.quant_layers[final_layer.name]
            if isinstance(out_val, list):
                return [dequantize(val, q_layer.out_scale, q_layer.out_zp) for val in out_val]
            else:
                return dequantize(out_val, q_layer.out_scale, q_layer.out_zp)
                
        return out_val

    def compare_layers(self, x, mode="quantize"):
        """
        Runs both float and quantized execution passes and returns a dictionary 
        containing comparison metrics (Max Diff, Mean Diff) for each layer.
        mode can be 'quantize' or 'integer'.
        """
        tensor_values_float = {}
        tensor_values_quant = {}
        
        # Feed inputs
        if not isinstance(x, dict):
            input_tensor = self.model.inputs[0]
            tensor_values_float[id(input_tensor)] = x
            tensor_values_quant[id(input_tensor)] = x
        else:
            for k, v in x.items():
                for inp in self.model.inputs:
                    if inp.name == k:
                        tensor_values_float[id(inp)] = v
                        tensor_values_quant[id(inp)] = v
                        break
                        
        comparison = {}
        
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                layer_output = layer.output
                q_layer = self.quant_layers[layer.name]
                if mode == "integer":
                    tensor_values_quant[id(layer_output)] = q_layer.quantized_infer_integer(x)
                else:
                    tensor_values_quant[id(layer_output)] = q_layer.quantized_infer(x)
                    
                if id(layer_output) not in tensor_values_float:
                    if not isinstance(x, dict):
                        tensor_values_float[id(layer_output)] = x
                continue
                
            q_layer = self.quant_layers[layer.name]
            
            # --- Float pass ---
            inputs_f = layer.input
            if isinstance(inputs_f, list):
                inp_vals_f = [tensor_values_float[id(t)] for t in inputs_f]
            else:
                inp_vals_f = tensor_values_float[id(inputs_f)]
                
            inputs_tf = [tf.convert_to_tensor(v, dtype=tf.float32) for v in inp_vals_f] if isinstance(inp_vals_f, list) else tf.convert_to_tensor(inp_vals_f, dtype=tf.float32)
            outputs_tf = layer(inputs_tf)
            out_val_f = outputs_tf.numpy()
            
            outputs_f = layer.output
            if isinstance(outputs_f, list):
                for j, out_t in enumerate(outputs_f):
                    tensor_values_float[id(out_t)] = out_val_f[j]
            else:
                tensor_values_float[id(outputs_f)] = out_val_f
                
            # --- Quantized pass ---
            inputs_q = layer.input
            if isinstance(inputs_q, list):
                inp_vals_q = [tensor_values_quant[id(t)] for t in inputs_q]
            else:
                inp_vals_q = tensor_values_quant[id(inputs_q)]
                
            if mode == "integer":
                out_val_q = q_layer.quantized_infer_integer(inp_vals_q)
            else:
                out_val_q = q_layer.quantized_infer(inp_vals_q)
                
            outputs_q = layer.output
            if isinstance(outputs_q, list):
                for j, out_t in enumerate(outputs_q):
                    tensor_values_quant[id(out_t)] = out_val_q[j]
            else:
                tensor_values_quant[id(outputs_q)] = out_val_q
                
            # --- Compute metrics ---
            if mode == "integer":
                if isinstance(out_val_q, list):
                    out_val_q_float = [dequantize(val, q_layer.out_scale, q_layer.out_zp) for val in out_val_q]
                else:
                    out_val_q_float = dequantize(out_val_q, q_layer.out_scale, q_layer.out_zp)
            else:
                out_val_q_float = out_val_q
                
            if isinstance(out_val_f, list):
                max_diff = np.max([np.max(np.abs(f - q)) for f, q in zip(out_val_f, out_val_q_float)])
                mean_diff = np.mean([np.mean(np.abs(f - q)) for f, q in zip(out_val_f, out_val_q_float)])
            else:
                max_diff = np.max(np.abs(out_val_f - out_val_q_float))
                mean_diff = np.mean(np.abs(out_val_f - out_val_q_float))
                
            comparison[layer.name] = {
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "layer_type": type(layer).__name__
            }
            
        return comparison
