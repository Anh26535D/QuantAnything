"""Loader and NumPy fallbacks for the C++ kernels.

The C++ source lives in ``kernels/qa_kernels.cpp`` and exposes a plain C ABI,
so it is loaded with :mod:`ctypes` -- there is no Python-header / pybind11
build dependency. On first import the library is compiled with the system C++
compiler and cached; if that is impossible (no compiler,
``QA_DISABLE_NATIVE=1``) every function silently falls back to an
equivalent pure NumPy implementation.

Environment variables:
    QA_DISABLE_NATIVE: set to ``1`` to force the NumPy fallbacks.
    QA_LIB: path of a pre-built ``libqa_kernels`` shared library to load.
    QA_MARCH: value for ``-march=`` when JIT compiling (default ``native``).
    CXX: C++ compiler to use.
"""

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_SRC = Path(__file__).parent / "kernels" / "qa_kernels.cpp"
_LIB_BASENAME = "libqa_kernels"

_lib = None
_load_error = None


# --------------------------------------------------------------- loading ----
def _lib_suffix():
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _cache_dir():
    candidates = [
        Path(__file__).parent / "kernels" / "_build",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "quantanything",
        Path(tempfile.gettempdir()) / "quantanything",
    ]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.access(path, os.W_OK):
                return path
        except OSError:
            continue
    raise OSError("no writable cache directory for the native kernels")


def _flag_sets():
    march = os.environ.get("QA_MARCH", "native")
    base = ["-O3", "-std=c++14", "-shared", "-fPIC", "-fvisibility=hidden"]
    sets = []
    for openmp in (["-fopenmp"], []):
        for arch in ([f"-march={march}"] if march else [], []):
            sets.append(base + arch + openmp)
    return sets


def build_library(force=False):
    """Compile the kernels and return the path of the shared library."""
    compiler = (
        os.environ.get("CXX")
        or shutil.which("g++")
        or shutil.which("c++")
        or shutil.which("clang++")
    )
    if compiler is None:
        raise RuntimeError("no C++ compiler found (set CXX)")
    source = _SRC.read_bytes()
    last_error = ""
    for flags in _flag_sets():
        digest = hashlib.sha256(
            source + " ".join(flags).encode() + platform.machine().encode()
        ).hexdigest()[:16]
        out = _cache_dir() / f"{_LIB_BASENAME}-{digest}{_lib_suffix()}"
        if out.exists() and not force:
            return out
        tmp = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
        cmd = [compiler, *flags, str(_SRC), "-o", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            os.replace(tmp, out)
            return out
        last_error = proc.stderr
        if tmp.exists():
            tmp.unlink()
    raise RuntimeError("failed to compile native kernels:\n" + last_error)


def _declare(lib):
    c = ctypes
    p, i64, f64 = c.c_void_p, c.c_int64, c.c_double
    sigs = {
        "qa_num_threads": (c.c_int, []),
        "qa_set_num_threads": (None, [c.c_int]),
        "qa_has_openmp": (c.c_int, []),
        "qa_gemm_f32": (None, [i64, i64, i64, p, p, p]),
        "qa_gemm_i16_i32": (None, [i64, i64, i64, p, p, p]),
        "qa_gemm_i32_i64": (None, [i64, i64, i64, p, p, p]),
        "qa_requantize_i32": (None, [p, p] + [i64] * 3 + [p] * 4 + [i64] * 3),
        "qa_requantize_i64": (None, [p, p] + [i64] * 3 + [p] * 4 + [i64] * 3),
        "qa_rescale": (None, [p, p] + [i64] * 7),
        "qa_add": (None, [p, p, p] + [i64] * 11),
        "qa_mul": (None, [p, p, p] + [i64] * 8),
        "qa_lut": (None, [p, p, i64, i64, p, i64]),
        "qa_quantize_f32": (None, [p, p, i64, f64, i64, i64, i64]),
        "qa_dequantize_f32": (None, [p, p, i64, f64, i64]),
    }
    for name in ("qa_conv_f32", "qa_conv_i32_i16_i32", "qa_conv_i32_i32_i64"):
        sigs[name] = (None, [p, p, p] + [i64] * 16)
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


def _load():
    global _lib, _load_error
    if os.environ.get("QA_DISABLE_NATIVE") == "1":
        _load_error = "disabled by QA_DISABLE_NATIVE"
        return
    try:
        path = os.environ.get("QA_LIB") or build_library()
        lib = ctypes.CDLL(str(path))
        _declare(lib)
        _lib = lib
    except Exception as exc:  # compiler missing, bad flags, ...
        _load_error = str(exc)


_load()


def available():
    """True when the C++ kernels are loaded (otherwise NumPy fallbacks run)."""
    return _lib is not None


def load_error():
    """Why the native kernels are not available (None when they are)."""
    return _load_error


def set_num_threads(n):
    """Set the OpenMP thread count of the native kernels (no-op without)."""
    if _lib is not None:
        _lib.qa_set_num_threads(int(n))


def _ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def _c(a, dtype):
    return np.ascontiguousarray(a, dtype=dtype)


# --------------------------------------------------- fixed-point multiply ----
def _as_i64(v):
    return np.asarray(v, dtype=np.int64)


def mbqm_np(x, mult, shift):
    """NumPy ``round(x * mult * 2**-(31+shift))`` with exact big-int fallback.

    ``mult`` / ``shift`` may be scalars or arrays broadcastable against ``x``.
    """
    x = _as_i64(x)
    mult = _as_i64(mult)
    ts = 31 + _as_i64(shift)
    xm = int(np.max(np.abs(x))) if x.size else 0
    mm = int(np.max(np.abs(mult))) if mult.size else 0
    if xm * mm < (1 << 62):
        product = x * mult
        right = ts > 0
        ts_r = np.where(right, ts, 1)
        rounded = (product + (np.int64(1) << (ts_r - 1))) >> ts_r
        left = product << np.where(right, 0, np.minimum(-ts, 60))
        return np.where(right, rounded, left).astype(np.int64)
    # Possible int64 overflow: slow but exact big-int path.
    xb, mb, tb = np.broadcast_arrays(x, mult, ts)
    out = np.empty(xb.shape, dtype=object)
    flat_x, flat_m, flat_t = xb.ravel(), mb.ravel(), tb.ravel()
    res = out.ravel()
    for i in range(flat_x.size):
        p = int(flat_x[i]) * int(flat_m[i])
        t = int(flat_t[i])
        res[i] = ((p + (1 << (t - 1))) >> t) if t > 0 else (p << -t)
    return np.clip(out, -(1 << 62), 1 << 62).astype(np.int64)


# ------------------------------------------------------------------ GEMM ----
def gemm_f32(a, b):
    """``a[M,K] @ b[K,N]`` in float32."""
    a, b = _c(a, np.float32), _c(b, np.float32)
    if _lib is None:
        return a @ b
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
    _lib.qa_gemm_f32(
        a.shape[0], b.shape[1], a.shape[1], _ptr(a), _ptr(b), _ptr(out)
    )
    return out


def gemm_int(a, b, wide=False):
    """Integer ``a[M,K] @ b[K,N]``; int32 accumulator, int64 when ``wide``."""
    m, k = a.shape
    n = b.shape[1]
    if _lib is None:
        return np.matmul(a.astype(np.int64), b.astype(np.int64)).astype(
            np.int64 if wide else np.int32
        )
    if wide:
        a, b = _c(a, np.int32), _c(b, np.int32)
        out = np.empty((m, n), dtype=np.int64)
        _lib.qa_gemm_i32_i64(m, n, k, _ptr(a), _ptr(b), _ptr(out))
    else:
        a, b = _c(a, np.int16), _c(b, np.int16)
        out = np.empty((m, n), dtype=np.int32)
        _lib.qa_gemm_i16_i32(m, n, k, _ptr(a), _ptr(b), _ptr(out))
    return out


# ----------------------------------------------------------- convolution ----
def conv_output_size(size, k, stride, pad_begin, pad_end, dilation):
    """Spatial output size of a convolution along one axis."""
    return (size + pad_begin + pad_end - dilation * (k - 1) - 1) // stride + 1


def _conv_geometry(x, w, strides, pads, dilations, groups):
    n, c, h, wd = x.shape
    cout, cg, kh, kw = w.shape
    if c != cg * groups or cout % groups:
        raise ValueError(
            f"conv shape mismatch: input C={c}, weight={w.shape}, "
            f"groups={groups}"
        )
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads
    oh = conv_output_size(h, kh, sh, pt, pb, dh)
    ow = conv_output_size(wd, kw, sw, pl, pr, dw)
    return n, c, h, wd, cout, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow


def _np_conv(x, w, strides, pads, dilations, groups, dtype):
    """Reference grouped convolution (im2col via sliding windows)."""
    n, c, h, wd, cout, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow = _conv_geometry(
        x, w, strides, pads, dilations, groups
    )
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pads[2]), (pl, pads[3])))
    ek_h, ek_w = dh * (kh - 1) + 1, dw * (kw - 1) + 1
    win = sliding_window_view(xp, (ek_h, ek_w), axis=(2, 3))
    win = win[:, :, ::sh, ::sw, ::dh, ::dw][:, :, :oh, :ow]
    cg, cog = c // groups, cout // groups
    out = np.empty((n, cout, oh, ow), dtype=dtype)
    for g in range(groups):
        wg = w[g * cog : (g + 1) * cog].astype(dtype)
        xg = win[:, g * cg : (g + 1) * cg].astype(dtype)
        out[:, g * cog : (g + 1) * cog] = np.einsum(
            "nchwyx,ocyx->nohw", xg, wg, optimize=True
        )
    return out


def conv2d_f32(
    x, w, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1), groups=1
):
    """Float32 NCHW convolution, weights ``[Cout, C/groups, KH, KW]``."""
    x, w = _c(x, np.float32), _c(w, np.float32)
    if _lib is None:
        return _np_conv(x, w, strides, pads, dilations, groups, np.float32)
    n, c, h, wd, cout, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow = _conv_geometry(
        x, w, strides, pads, dilations, groups
    )
    out = np.empty((n, cout, oh, ow), dtype=np.float32)
    _lib.qa_conv_f32(
        _ptr(x),
        _ptr(w),
        _ptr(out),
        n,
        c,
        h,
        wd,
        cout,
        kh,
        kw,
        sh,
        sw,
        pt,
        pl,
        dh,
        dw,
        groups,
        oh,
        ow,
    )
    return out


def conv2d_int(
    x,
    w,
    strides=(1, 1),
    pads=(0, 0, 0, 0),
    dilations=(1, 1),
    groups=1,
    wide=False,
):
    """Integer NCHW convolution of *zero-point centred* operands.

    Args:
        x: ``[N, C, H, W]`` integer activations minus their zero point.
        w: ``[Cout, C/groups, KH, KW]`` integer weights minus their zero point.
        wide: accumulate in int64 (int32 operands) instead of int32 (int16
            operands). The caller is responsible for choosing a path that
            cannot overflow, see :func:`accumulator_is_wide`.

    Returns:
        ``[N, Cout, OH, OW]`` accumulators (int32, or int64 when ``wide``).
    """
    if _lib is None:
        out = _np_conv(
            x.astype(np.int64),
            w.astype(np.int64),
            strides,
            pads,
            dilations,
            groups,
            np.int64,
        )
        return out if wide else out.astype(np.int32)
    x = _c(x, np.int32)
    n, c, h, wd, cout, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow = _conv_geometry(
        x, w, strides, pads, dilations, groups
    )
    args = (n, c, h, wd, cout, kh, kw, sh, sw, pt, pl, dh, dw, groups, oh, ow)
    if wide:
        w = _c(w, np.int32)
        out = np.empty((n, cout, oh, ow), dtype=np.int64)
        _lib.qa_conv_i32_i32_i64(_ptr(x), _ptr(w), _ptr(out), *args)
    else:
        w = _c(w, np.int16)
        out = np.empty((n, cout, oh, ow), dtype=np.int32)
        _lib.qa_conv_i32_i16_i32(_ptr(x), _ptr(w), _ptr(out), *args)
    return out


def accumulator_is_wide(num_bits, reduction_size):
    """Whether an integer dot product needs the int64 accumulator path.

    The int32 path stores operands as int16, so it is only valid when the
    zero-point-centred operands fit int16 (``num_bits <= 15``) and
    ``|a| * |w| * K`` provably stays below 2**31.
    """
    if num_bits > 15:
        return True
    bound = ((1 << num_bits) ** 2) * int(reduction_size)
    return bound >= (1 << 31)


# -------------------------------------------------------- requantisation ----
def requantize(
    acc,
    outer,
    channels,
    inner,
    mult,
    shift,
    zp,
    qmin,
    qmax,
    pre_bias=None,
    post_bias=None,
):
    """Per-channel fixed-point requantisation of an accumulator tensor.

    ``acc`` is viewed as ``[outer, channels, inner]`` and every channel ``c``
    is mapped to
    ``clip(mbqm(acc + pre_bias[c], mult[c], shift[c]) + post_bias[c] + zp)``.

    Returns:
        An int32 array with the same shape as ``acc``.
    """
    mult = np.broadcast_to(_as_i64(mult), (channels,))
    shift = np.broadcast_to(_as_i64(shift), (channels,))
    if _lib is None:
        a = acc.reshape(outer, channels, inner).astype(np.int64)
        if pre_bias is not None:
            a = a + _as_i64(pre_bias).reshape(1, channels, 1)
        r = mbqm_np(
            a, mult.reshape(1, channels, 1), shift.reshape(1, channels, 1)
        )
        if post_bias is not None:
            r = r + _as_i64(post_bias).reshape(1, channels, 1)
        r = np.clip(r + int(zp), qmin, qmax).astype(np.int32)
        return r.reshape(acc.shape)
    wide = acc.dtype == np.int64
    acc = _c(acc, np.int64 if wide else np.int32)
    out = np.empty(acc.shape, dtype=np.int32)
    mult, shift = _c(mult, np.int64), _c(shift, np.int64)
    pre = (
        None
        if pre_bias is None
        else _c(np.broadcast_to(_as_i64(pre_bias), (channels,)), np.int64)
    )
    post = (
        None
        if post_bias is None
        else _c(np.broadcast_to(_as_i64(post_bias), (channels,)), np.int64)
    )
    fn = _lib.qa_requantize_i64 if wide else _lib.qa_requantize_i32
    fn(
        _ptr(acc),
        _ptr(out),
        outer,
        channels,
        inner,
        _ptr(mult),
        _ptr(shift),
        _ptr(pre) if pre is not None else None,
        _ptr(post) if post is not None else None,
        int(zp),
        int(qmin),
        int(qmax),
    )
    return out


def rescale(x, zp_in, mult, shift, zp_out, qmin, qmax):
    """``clip(mbqm(x - zp_in) + zp_out)`` element-wise."""
    if _lib is None:
        r = mbqm_np(_as_i64(x) - int(zp_in), mult, shift) + int(zp_out)
        return np.clip(r, qmin, qmax).astype(np.int32)
    x = _c(x, np.int32)
    out = np.empty_like(x)
    _lib.qa_rescale(
        _ptr(x),
        _ptr(out),
        x.size,
        int(zp_in),
        int(mult),
        int(shift),
        int(zp_out),
        int(qmin),
        int(qmax),
    )
    return out


def add(a, b, za, ma, sa, zb, mb, sb, zp_out, qmin, qmax, sign=1):
    """Rescale both operands to the output scale and add (or subtract)."""
    a, b = np.broadcast_arrays(a, b)
    if _lib is None:
        r = (
            mbqm_np(_as_i64(a) - int(za), ma, sa)
            + sign * mbqm_np(_as_i64(b) - int(zb), mb, sb)
            + int(zp_out)
        )
        return np.clip(r, qmin, qmax).astype(np.int32)
    a, b = _c(a, np.int32), _c(b, np.int32)
    out = np.empty_like(a)
    _lib.qa_add(
        _ptr(a),
        _ptr(b),
        _ptr(out),
        a.size,
        int(za),
        int(ma),
        int(sa),
        int(zb),
        int(mb),
        int(sb),
        int(sign),
        int(zp_out),
        int(qmin),
        int(qmax),
    )
    return out


def mul(a, b, za, zb, mult, shift, zp_out, qmin, qmax):
    """``clip(mbqm((a - za) * (b - zb)) + zp_out)`` element-wise."""
    a, b = np.broadcast_arrays(a, b)
    if _lib is None:
        prod = (_as_i64(a) - int(za)) * (_as_i64(b) - int(zb))
        r = mbqm_np(prod, mult, shift) + int(zp_out)
        return np.clip(r, qmin, qmax).astype(np.int32)
    a, b = _c(a, np.int32), _c(b, np.int32)
    out = np.empty_like(a)
    _lib.qa_mul(
        _ptr(a),
        _ptr(b),
        _ptr(out),
        a.size,
        int(za),
        int(zb),
        int(mult),
        int(shift),
        int(zp_out),
        int(qmin),
        int(qmax),
    )
    return out


def lut(x, table, offset):
    """``table[clip(x - offset, 0, len(table) - 1)]``."""
    table = _c(table, np.int32)
    if _lib is None:
        idx = np.clip(_as_i64(x) - int(offset), 0, len(table) - 1)
        return table[idx]
    x = _c(x, np.int32)
    out = np.empty_like(x)
    _lib.qa_lut(
        _ptr(x), _ptr(out), x.size, int(offset), _ptr(table), len(table)
    )
    return out


# ------------------------------------------------- float <-> integer ----
def quantize_f32(x, scale, zp, qmin, qmax):
    """Per-tensor quantisation of float data to int32 (half-to-even)."""
    x = _c(x, np.float32)
    if _lib is None:
        q = np.round(x.astype(np.float64) / scale) + zp
        return np.clip(q, qmin, qmax).astype(np.int32)
    out = np.empty(x.shape, dtype=np.int32)
    _lib.qa_quantize_f32(
        _ptr(x), _ptr(out), x.size, float(scale), int(zp), int(qmin), int(qmax)
    )
    return out


def dequantize_f32(q, scale, zp):
    """Per-tensor dequantisation of int32 data to float32."""
    q = _c(q, np.int32)
    if _lib is None:
        return ((q.astype(np.int64) - zp) * float(scale)).astype(np.float32)
    out = np.empty(q.shape, dtype=np.float32)
    _lib.qa_dequantize_f32(_ptr(q), _ptr(out), q.size, float(scale), int(zp))
    return out
