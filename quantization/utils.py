import numpy as np

def parse_quant_type(quant_type):
    """
    Parses a quantization type string (e.g. 'int8', 'uint8', 'int9') into bitwidth and signedness.
    """
    if quant_type.startswith('uint'):
        signed = False
        num_bits = int(quant_type[4:])
    elif quant_type.startswith('int'):
        signed = True
        num_bits = int(quant_type[3:])
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")
    return num_bits, signed

def calculate_scale_zp(min_val, max_val, quant_type="int8"):
    """
    Calculates quantization scale and zero point for a given min/max range and quantization type.
    """
    num_bits, signed = parse_quant_type(quant_type)
    
    if signed:
        # Symmetric signed quantization
        qmin = -(2**(num_bits - 1))
        qmax = 2**(num_bits - 1) - 1
        
        max_abs = max(abs(min_val), abs(max_val))
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / qmax
        zp = 0
    else:
        # Asymmetric unsigned quantization
        qmin = 0
        qmax = 2**num_bits - 1
        
        if min_val == max_val:
            scale = 1.0
            zp = 0
        else:
            scale = (max_val - min_val) / qmax
            zp = int(np.round(-min_val / scale))
            zp = np.clip(zp, qmin, qmax)
            
    return scale, zp, qmin, qmax

def quantize(x, scale, zp, qmin, qmax):
    """
    Quantizes float input tensor into integer space.
    Supports both scalar and numpy array scales (per-channel).
    """
    if isinstance(scale, np.ndarray):
        # Avoid division by zero by replacing zero scale values with 1.0 (will zero out afterward)
        scale_safe = np.where(scale == 0, 1.0, scale)
        q = np.round(x / scale_safe) + zp
        # Wherever scale is 0, output should be zp (corresponding to quantized 0.0)
        # Use np.where with broadcasted comparison if shapes differ
        q = np.where(scale == 0, zp, q)
    else:
        if scale == 0:
            return np.zeros_like(x)
        q = np.round(x / scale) + zp
    return np.clip(q, qmin, qmax)

def dequantize(q, scale, zp):
    """
    Dequantizes integer input back to float space.
    """
    return (q - zp) * scale

def fake_quantize(x, scale, zp, qmin, qmax):
    """
    Simulates quantization effects on float inputs (fake quantization).
    """
    q = quantize(x, scale, zp, qmin, qmax)
    return dequantize(q, scale, zp)
