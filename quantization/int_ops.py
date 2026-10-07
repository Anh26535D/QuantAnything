"""Integer-only softmax and layer normalization.

Only integer additions, multiplications, shifts and comparisons are used. The
non-linear parts are replaced by piecewise-linear functions with at most four
segments (see :mod:`quantization.pwl`):

* ``exp``: ``e^-x = 2^-(x log2 e) = 2^-n * 2^-f`` with ``n`` the integer and
  ``f`` the fractional part (Q16): ``2^-n`` is a right shift, ``2^-f`` on
  ``[0, 1)`` a 4-segment PWL,
* ``1 / sum``: the sum is normalized to a mantissa in ``[1, 2)`` (leading-bit
  position + shift) and ``1/m`` is a 4-segment PWL,
* ``1 / sqrt(var)``: the variance is normalized to a mantissa in ``[1, 4)``
  with an *even* exponent, so ``2^(-e/2)`` is again a shift, and ``1/sqrt(m)``
  is a 4-segment PWL.
"""

import functools

import numpy as np

from quantization import pwl
from quantization.utils import multiply_by_quantized_multiplier as mbqm
from quantization.utils import quantize_multiplier

LOG2_E = 1.4426950408889634
EXP_FRAC_BITS = 16  # fraction of the exponent argument (Q16)
EXP_OUT_BITS = 15  # 2^-f in Q15
MANT_BITS = 15  # reciprocal mantissa Q15
RSQRT_IN_BITS = 14  # rsqrt mantissa Q14
VAR_FRAC_BITS = 8  # variance fixed-point fraction
CENTER_FRAC_BITS = 4  # fraction of the centred values in LayerNorm
LN_INPUT_BITS = 18  # magnitude bits of the LayerNorm input that are kept


@functools.lru_cache(maxsize=None)
def exp2_neg():
    """4-segment PWL of ``2^-u`` for ``u`` in ``[0, 1]`` (Q16 -> Q15)."""
    return pwl.fixed_point(
        lambda u: 2.0**-u, 0.0, 1.0, EXP_FRAC_BITS, EXP_OUT_BITS
    )


@functools.lru_cache(maxsize=None)
def reciprocal():
    """4-segment PWL of ``1/m`` for ``m`` in ``[1, 2]`` (Q15 -> Q15)."""
    return pwl.fixed_point(lambda m: 1.0 / m, 1.0, 2.0, MANT_BITS, MANT_BITS)


@functools.lru_cache(maxsize=None)
def rsqrt():
    """4-segment PWL of ``1/sqrt(m)`` for ``m`` in ``[1, 4]`` (Q14 -> Q15)."""
    return pwl.fixed_point(
        lambda m: 1.0 / np.sqrt(m), 1.0, 4.0, RSQRT_IN_BITS, MANT_BITS
    )


def ilog2(v):
    """``floor(log2(v))`` of positive int64 values (exact below 2**53)."""
    return np.frexp(np.asarray(v, dtype=np.float64))[1].astype(np.int64) - 1


def _normalize(v, bits):
    """Shifts ``v`` so that ``v / 2**L`` lies in ``[2**bits, 2**(bits+1))``."""
    exp = ilog2(v)
    shr = np.maximum(exp - bits, 0)
    shl = np.maximum(bits - exp, 0)
    return (v << shl) >> shr, exp


def softmax(codes, in_scale, out_scale, out_zp, qmin, qmax):
    """Integer softmax over the last axis.

    Args:
        codes: integer logits; the real value is ``code * in_scale`` (any
            constant offset cancels in the softmax).
        in_scale: scale of the logits.
        out_scale, out_zp, qmin, qmax: output grid
            (``real = (q - zp) * scale``).

    Returns:
        int32 probability codes.
    """
    x = np.asarray(codes, dtype=np.int64)
    d = x.max(axis=-1, keepdims=True) - x  # >= 0
    m_t, s_t = quantize_multiplier(in_scale * LOG2_E * (1 << EXP_FRAC_BITS))
    t = mbqm(d, m_t, s_t)  # d*log2e, Q16
    n = t >> EXP_FRAC_BITS
    frac = (t & ((1 << EXP_FRAC_BITS) - 1)).astype(np.int32)
    g = exp2_neg()[0](frac).astype(np.int64)  # 2^-f, Q15
    e = g >> np.minimum(n, 40)  # 2^-n * 2^-f
    total = e.sum(axis=-1, keepdims=True)  # >= 2^15 * ~1
    mant, exp = _normalize(total, MANT_BITS)  # [1,2) in Q15
    r = reciprocal()[0](mant.astype(np.int32)).astype(np.int64)
    prod = e * r  # e / S * 2^(15+L)
    m_k, s_k = quantize_multiplier(1.0 / out_scale)
    out = mbqm(prod, m_k, s_k + exp + MANT_BITS) + int(out_zp)
    return np.clip(out, qmin, qmax).astype(np.int32)


def layer_norm(
    codes, in_scale, eps, mult, shift, post_bias, out_zp, qmin, qmax
):
    """Integer LayerNorm over the last axis (``gamma``, ``beta`` per channel).

    The output code is ``clip(round(y / out_scale) + out_zp)`` with
    ``y = (x - mean) / sqrt(var + eps) * gamma + beta``; ``mult`` / ``shift``
    / ``post_bias`` encode ``gamma / out_scale * 2**-15`` and
    ``beta / out_scale`` per channel (see :func:`prepare_layer_norm`).
    """
    x = np.asarray(codes, dtype=np.int64)
    # Wide inputs (e.g. a 24-bit residual stream): keep the top LN_INPUT_BITS
    # bits (block-floating-point style) so the squares stay inside int64.
    mag = int(np.abs(x).max()) if x.size else 0
    drop = max(0, mag.bit_length() - LN_INPUT_BITS)
    if drop:
        x = (x + (1 << (drop - 1))) >> drop
        in_scale = in_scale * (1 << drop)
    c = x.shape[-1]
    m_c, s_c = quantize_multiplier(1.0 / c)
    q = CENTER_FRAC_BITS
    mean_q = mbqm(x.sum(axis=-1, keepdims=True) << q, m_c, s_c)  # Q4
    d = (x << q) - mean_q  # Q4, centred
    var_q = mbqm((d * d).sum(axis=-1, keepdims=True), m_c, s_c)  # Q8 code^2
    eps_q = max(
        1, int(round(eps / (in_scale * in_scale) * (1 << VAR_FRAC_BITS)))
    )
    v = var_q + eps_q
    exp = ilog2(v)
    e2 = exp - (exp & 1)  # even exponent
    shr = np.maximum(e2 - RSQRT_IN_BITS, 0)
    shl = np.maximum(RSQRT_IN_BITS - e2, 0)
    mant = ((v << shl) >> shr).astype(np.int32)  # [1,4) in Q14
    r = rsqrt()[0](mant).astype(np.int64)  # Q15
    half = e2 >> 1
    acc = d * r  # Q4 * Q15
    acc = (acc + (1 << np.maximum(half - 1, 0)) * (half > 0)) >> half
    # acc = (x - mean) / std in Q15
    flat = acc.reshape(-1, c).astype(np.int64)
    from quantization import native

    res = native.requantize(
        flat,
        flat.shape[0],
        c,
        1,
        mult,
        shift,
        out_zp,
        qmin,
        qmax,
        post_bias=post_bias,
    )
    return res.reshape(x.shape)


def prepare_layer_norm(gamma, beta, out_scale):
    """Per-channel ``(mult, shift, post_bias)`` for :func:`layer_norm`."""
    from quantization.utils import quantize_multipliers

    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    mult, shift = quantize_multipliers(gamma / (out_scale * (1 << MANT_BITS)))
    post = np.round(beta / out_scale).astype(np.int64)
    return mult, shift, post


# ------------------------------------------- float emulation (fake quant) ----
def softmax_float(logits, axis=-1):
    """Float softmax that uses the same 4-segment PWLs as :func:`softmax`."""
    x = np.moveaxis(np.asarray(logits, dtype=np.float64), axis, -1)
    t = (x.max(axis=-1, keepdims=True) - x) * LOG2_E
    n = np.floor(t)
    e = exp2_neg()[1](t - n) * np.exp2(-n)
    total = e.sum(axis=-1, keepdims=True)
    mant, exp = np.frexp(total)  # mant in [0.5, 1)
    p = e * reciprocal()[1](2.0 * mant) * np.exp2(-(exp - 1.0))
    return np.moveaxis(p, -1, axis)


def layer_norm_float(x, gamma, beta, eps):
    """Float LayerNorm (last axis) with the 4-segment ``1/sqrt`` PWL."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    v = ((x - mean) ** 2).mean(axis=-1, keepdims=True) + eps
    exp = np.floor(np.log2(v))
    e2 = exp - np.mod(exp, 2)  # even exponent
    r = rsqrt()[1](v * np.exp2(-e2)) * np.exp2(-e2 / 2.0)
    return (x - mean) * r * gamma + beta
