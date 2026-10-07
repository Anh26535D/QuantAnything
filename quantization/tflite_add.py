"""Bit-exact port of TensorFlow Lite's quantized ``Add`` (left-shift add).

Reference: ``tensorflow/lite/kernels/add.cc`` (parameter preparation),
``kernels/internal/reference/{add.h,integer_ops/add.h}`` (the element-wise
kernels) and gemmlowp's ``fixedpoint.h`` (``SaturatingRoundingDoublingHighMul``
and ``RoundingDivideByPOT``). The golden vectors in
``tests/data/tflite_add_golden.npz`` were produced by compiling Google's own
code, see ``tests/test_tflite_add.py``.

General path (int8: ``left_shift = 20``, int16: ``left_shift = 15``)::

    v_k  = (x_k - zp_k) << left_shift
    s_k  = MBQM(v_k, m_k, sh_k)       m_k ~ scale_k / (2 * max(scale_1, scale_2))
    out  = MBQM(s_1 + s_2, m_o, sh_o) + zp_o      m_o ~ 2 * max / (2^ls * scale_o)

Power-of-two int16 path (zero zero-points, power-of-two scales): no
multipliers and no left shift; one operand already has the output scale, the
other is right-shifted (``RoundingDivideByPOT``) and the two are added with
saturation.
"""

import math
from dataclasses import dataclass

import numpy as np

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def srdhm(a, b):
    """gemmlowp ``SaturatingRoundingDoublingHighMul`` on int32 values."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    overflow = (a == b) & (a == INT32_MIN)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    num = ab + nudge
    # C++ integer division truncates toward zero
    q = np.where(num >= 0, num >> 31, -((-num) >> 31))
    return np.where(overflow, INT32_MAX, q)


def rounding_divide_by_pot(x, exponent):
    """gemmlowp ``RoundingDivideByPOT`` (round half away from zero)."""
    x = np.asarray(x, dtype=np.int64)
    if not 0 <= exponent <= 31:
        raise ValueError("exponent must be in 0..31")
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> exponent) + (remainder > threshold)


def mbqm_smaller_than_one(x, multiplier, left_shift):
    """``MultiplyByQuantizedMultiplierSmallerThanOneExp`` (double rounding)."""
    return rounding_divide_by_pot(srdhm(x, multiplier), -left_shift)


def quantize_multiplier(real):
    """``QuantizeMultiplier``: ``real ~ q * 2**(shift - 31)``, ``q`` int32."""
    if real == 0.0:
        return 0, 0
    frac, shift = math.frexp(real)
    q = int(math.floor(frac * (1 << 31) + 0.5))  # std::round, positive
    if q == 1 << 31:
        q //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q, shift


def quantize_multiplier_smaller_than_one(real):
    """``(q, left_shift)`` with ``left_shift <= 0`` for ``0 < real < 1``."""
    if not 0.0 < real < 1.0:
        raise ValueError(f"multiplier {real} is not in (0, 1)")
    q, shift = quantize_multiplier(real)
    if shift > 0:
        raise ValueError("multiplier needs a positive shift")
    return q, shift


@dataclass
class AddParams:
    """Everything ``add.cc::Prepare`` derives for the general path."""

    left_shift: int
    m1: int
    sh1: int
    m2: int
    sh2: int
    mo: int
    sho: int
    offset1: int
    offset2: int
    offset_out: int
    qmin: int
    qmax: int


def prepare(s1, s2, s_out, zp1, zp2, zp_out, qmin, qmax, left_shift=None):
    """General-path parameters (``left_shift``: 20 for int8, 15 for int16)."""
    if left_shift is None:
        left_shift = 15 if qmax - qmin >= 2**15 else 20
    twice_max = 2.0 * max(s1, s2)
    m1, sh1 = quantize_multiplier_smaller_than_one(s1 / twice_max)
    m2, sh2 = quantize_multiplier_smaller_than_one(s2 / twice_max)
    mo, sho = quantize_multiplier_smaller_than_one(
        twice_max / ((1 << left_shift) * s_out)
    )
    return AddParams(
        left_shift,
        m1,
        sh1,
        m2,
        sh2,
        mo,
        sho,
        -int(zp1),
        -int(zp2),
        int(zp_out),
        int(qmin),
        int(qmax),
    )


def add(x1, x2, p):
    """Element-wise left-shift add; raises if ``x << left_shift`` overflows."""
    a = np.asarray(x1, dtype=np.int64) + p.offset1
    b = np.asarray(x2, dtype=np.int64) + p.offset2
    a = a << p.left_shift
    b = b << p.left_shift
    if max(np.abs(a).max(initial=0), np.abs(b).max(initial=0)) > INT32_MAX:
        raise OverflowError(
            "operand << left_shift does not fit int32: use a smaller "
            "left_shift or a narrower operand grid"
        )
    sa = mbqm_smaller_than_one(a, p.m1, p.sh1)
    sb = mbqm_smaller_than_one(b, p.m2, p.sh2)
    raw = mbqm_smaller_than_one(sa + sb, p.mo, p.sho) + p.offset_out
    return np.clip(raw, p.qmin, p.qmax).astype(np.int32)


def pot_shifts(s1, s2, s_out):
    """``(shift1, shift2)`` of the power-of-two path or ``None``.

    Requires power-of-two scales with one operand on the output scale and the
    other not coarser than it (``shift <= 0``).
    """
    logs = []
    for s in (s1, s2, s_out):
        e = math.log2(s)
        if abs(e - round(e)) > 1e-9:
            return None
        logs.append(int(round(e)))
    sh1, sh2 = logs[0] - logs[2], logs[1] - logs[2]
    if not ((sh1 == 0 or sh2 == 0) and sh1 <= 0 and sh2 <= 0):
        return None
    return sh1, sh2


def add_pot(x1, x2, shifts, qmin=-(2**15), qmax=2**15 - 1):
    """Power-of-two int16 add: right shift one operand, saturating add."""
    sh1, sh2 = shifts
    x1 = np.asarray(x1, dtype=np.int64)
    x2 = np.asarray(x2, dtype=np.int64)
    if sh1 == 0:
        keep, moved, right = x1, x2, -sh2
    else:
        keep, moved, right = x2, x1, -sh1
    total = keep + rounding_divide_by_pot(moved, right)
    total = np.clip(total, -(2**15), 2**15 - 1)  # SaturatingAdd (int16)
    return np.clip(total, qmin, qmax).astype(np.int32)
