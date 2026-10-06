"""Low-level quantization maths (NumPy, C++ accelerated when available)."""

from dataclasses import dataclass

import numpy as np

from quantization import native

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1


@dataclass
class QParams:
    """Affine quantization parameters of one tensor.

    ``real = (q - zp) * scale`` with ``qmin <= q <= qmax``.
    """

    scale: float
    zp: int
    qmin: int
    qmax: int

    def to_dict(self):
        """JSON-serializable form."""
        return {
            "scale": float(self.scale),
            "zp": int(self.zp),
            "qmin": int(self.qmin),
            "qmax": int(self.qmax),
        }

    @classmethod
    def from_dict(cls, d):
        """Inverse of :meth:`to_dict`."""
        return cls(
            float(d["scale"]), int(d["zp"]), int(d["qmin"]), int(d["qmax"])
        )


def parse_quant_type(quant_type):
    """Parses a type string such as ``'int8'`` / ``'uint16'``.

    Returns:
        ``(num_bits, signed)``.
    """
    if quant_type.startswith("uint"):
        signed = False
        num_bits = int(quant_type[4:])
    elif quant_type.startswith("int"):
        signed = True
        num_bits = int(quant_type[3:])
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")
    return num_bits, signed


def quant_range(quant_type):
    """``(qmin, qmax)`` of a quantization type string."""
    num_bits, signed = parse_quant_type(quant_type)
    if signed:
        return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return 0, 2**num_bits - 1


def calculate_scale_zp(min_val, max_val, quant_type="int8"):
    """Scale / zero point for a calibrated ``[min_val, max_val]`` range.

    Signed types use symmetric quantization (``zp = 0``); unsigned types use
    asymmetric quantization over a range that always contains 0.

    Returns:
        ``(scale, zp, qmin, qmax)``.
    """
    qmin, qmax = quant_range(quant_type)
    _, signed = parse_quant_type(quant_type)
    min_val, max_val = float(min_val), float(max_val)

    if signed:
        max_abs = max(abs(min_val), abs(max_val))
        scale = 1.0 if max_abs == 0 else max_abs / qmax
        zp = 0
    else:
        min_val, max_val = min(min_val, 0.0), max(max_val, 0.0)
        if min_val == max_val:
            scale, zp = 1.0, 0
        else:
            scale = (max_val - min_val) / qmax
            zp = int(np.clip(np.round(-min_val / scale), qmin, qmax))
    return scale, zp, qmin, qmax


def _int_dtype(qmin, qmax):
    return np.int32 if _INT32_MIN <= qmin and qmax <= _INT32_MAX else np.int64


def quantize(x, scale, zp, qmin, qmax):
    """Float -> integer (round-half-to-even, clipped to ``[qmin, qmax]``).

    ``scale`` / ``zp`` may be per-channel arrays broadcastable against ``x``.
    Returns int32 (int64 when the range does not fit).
    """
    x = np.asarray(x)
    dtype = _int_dtype(qmin, qmax)
    if np.ndim(scale) == 0 and np.ndim(zp) == 0:
        if scale == 0:
            return np.full(x.shape, zp, dtype=dtype)
        if dtype == np.int32 and x.dtype in (np.float32, np.float64):
            return native.quantize_f32(x, scale, zp, qmin, qmax)
    scale = np.asarray(scale, dtype=np.float64)
    safe = np.where(scale == 0, 1.0, scale)
    q = np.round(x.astype(np.float64) / safe) + zp
    q = np.where(scale == 0, zp, q)
    return np.clip(q, qmin, qmax).astype(dtype)


def dequantize(q, scale, zp):
    """Integer -> float32: ``(q - zp) * scale``."""
    q = np.asarray(q)
    if np.ndim(scale) == 0 and np.ndim(zp) == 0 and q.dtype == np.int32:
        return native.dequantize_f32(q, scale, zp)
    return (
        (q.astype(np.int64) - zp) * np.asarray(scale, dtype=np.float64)
    ).astype(np.float32)


def fake_quantize(x, scale, zp, qmin, qmax):
    """Simulates quantization noise on float data (quantize + dequantize)."""
    return dequantize(quantize(x, scale, zp, qmin, qmax), scale, zp)


def quantize_multiplier(real_multiplier):
    """Float multiplier -> ``(M0, shift)`` with ``real = M0 * 2**-(31+shift)``.

    ``|M0|`` lies in ``[2**30, 2**31)`` and carries the sign of the input
    (negative multipliers occur e.g. for negative BatchNorm gamma); zero
    returns ``(0, 0)``.
    """
    if real_multiplier < 0:
        m, s = quantize_multiplier(-real_multiplier)
        return -m, s
    if real_multiplier == 0:
        return 0, 0
    significand, exponent = np.frexp(real_multiplier)
    q_multiplier = int(np.round(significand * (1 << 31)))
    if q_multiplier >= (1 << 31):
        q_multiplier //= 2
        exponent += 1
    return q_multiplier, int(-exponent)


def quantize_multipliers(real_multipliers):
    """Vectorised :func:`quantize_multiplier` -> two int64 arrays."""
    flat = np.asarray(real_multipliers, dtype=np.float64).reshape(-1)
    pairs = [quantize_multiplier(float(m)) for m in flat]
    mult = np.array([p[0] for p in pairs], dtype=np.int64)
    shift = np.array([p[1] for p in pairs], dtype=np.int64)
    shape = np.shape(real_multipliers)
    return mult.reshape(shape), shift.reshape(shape)


def multiply_by_quantized_multiplier(accum, q_multiplier, shift):
    """``round(accum * q_multiplier * 2**-(31+shift))`` in fixed point.

    Exact even when the 64-bit product would overflow.
    """
    return native.mbqm_np(accum, q_multiplier, shift)
