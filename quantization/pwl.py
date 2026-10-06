"""Piecewise-linear (PWL) approximation of non-linear functions.

Every non-linear function of the integer pipeline (GELU, SiLU, sigmoid,
``2^-x`` inside softmax, ``1/x``, ``1/sqrt(x)`` ...) is replaced by a
continuous PWL function with **at most four segments** that is evaluated with
integer arithmetic only:

``y = y_i + mbqm(x - x_i, M_i, shift_i)``  for  ``x_i <= x < x_{i+1}``

where ``(M_i, shift_i)`` is the fixed-point representation of the segment
slope. The knots are found offline: dynamic programming on chord errors gives
good break points and a Lawson iteration then refines the knot values to the
minimax (or minimax-relative) solution.
"""

from dataclasses import dataclass

import numpy as np

from quantization import native
from quantization.utils import quantize_multiplier

MAX_SEGMENTS = 4


# ---------------------------------------------------------------- fitting ----
@dataclass
class PWL:
    """Continuous piecewise-linear function given by its knots."""

    x: np.ndarray  # segments + 1 knot positions
    y: np.ndarray  # knot values
    max_error: float = 0.0  # on the fitted domain (absolute or relative)
    relative: bool = False

    @property
    def segments(self):
        """Number of linear segments."""
        return len(self.x) - 1

    def __call__(self, x):
        """Evaluates the function (end segments extend linearly)."""
        x = np.asarray(x, dtype=np.float64)
        idx = np.clip(
            np.searchsorted(self.x, x, side="right") - 1, 0, self.segments - 1
        )
        slope = np.diff(self.y) / np.diff(self.x)
        return self.y[idx] + slope[idx] * (x - self.x[idx])


def _hat_basis(knots, xs):
    """Matrix ``A[s, k]`` of hat functions: ``A @ y_knots`` interpolates."""
    idx = np.clip(
        np.searchsorted(knots, xs, side="right") - 1, 0, len(knots) - 2
    )
    t = (xs - knots[idx]) / (knots[idx + 1] - knots[idx])
    a = np.zeros((len(xs), len(knots)))
    rows = np.arange(len(xs))
    a[rows, idx] = 1.0 - t
    a[rows, idx + 1] = t
    return a


def _lawson(a, target, weight, iters=80):
    """Minimax fit of ``a @ y`` to ``target`` (weighted), Lawson iteration."""
    omega = np.ones(len(target)) / len(target)
    best, best_err = None, np.inf
    for _ in range(iters):
        sw = np.sqrt(omega) * weight
        y, *_ = np.linalg.lstsq(a * sw[:, None], target * sw, rcond=None)
        err = np.abs(weight * (a @ y - target))
        if err.max() < best_err:
            best, best_err = y, err.max()
        omega = omega * err
        total = omega.sum()
        if total <= 0:
            break
        omega /= total
    return best, best_err


def _exact_pwl(f, lo, hi, segments, n=4097):
    """The function itself when it already is piecewise linear.

    Finds the kinks from the second differences and intersects the adjacent
    linear pieces; returns ``None`` unless at most ``segments`` pieces
    reproduce ``f`` to machine precision.
    """
    x = np.linspace(lo, hi, n)
    y = np.asarray(f(x), dtype=np.float64)
    h = x[1] - x[0]
    slope = np.diff(y) / h
    scale = max(float(np.abs(y).max()), 1.0)
    # float32 evaluation noise (~1e-7 relative) must not look like a kink
    jump = np.abs(np.diff(slope)) > 1e-5 * scale / h
    cells = np.flatnonzero(jump)  # kink inside cell cells + 1
    groups = []
    for c in cells:
        if groups and c - groups[-1][-1] <= 1:
            groups[-1].append(c)
        else:
            groups.append([c])
    if len(groups) > segments - 1:
        return None
    knots = [lo]
    for g in groups:
        left, right = g[0], g[-1] + 1  # slope index before / after
        s0, s1 = slope[left], slope[right]
        x0, y0 = x[left], y[left]
        x1, y1 = x[right + 1], y[right + 1]
        # intersect  y0 + s0 (t - x0)  with  y1 + s1 (t - x1)
        t = (y1 - y0 + s0 * x0 - s1 * x1) / (s0 - s1)
        knots.append(float(t))
    knots.append(hi)
    knots = np.array(knots)
    if np.any(np.diff(knots) <= 1e-12):
        return None
    vals = np.asarray(f(knots), dtype=np.float64)
    cand = PWL(knots, vals, 0.0)
    xs = np.linspace(lo, hi, 9973)
    err = float(np.abs(cand(xs) - np.asarray(f(xs), dtype=np.float64)).max())
    if err > 2e-5 * scale:
        return None
    cand.max_error = err
    return cand


def fit_pwl(
    f, lo, hi, segments=MAX_SEGMENTS, relative=False, grid=129, samples=None
):
    """Fits a continuous PWL function with ``<= 4`` segments to ``f``.

    Args:
        f: vectorized callable on float64 arrays.
        lo, hi: fitted domain.
        segments: number of linear segments (at most 4).
        relative: minimize the *relative* error ``|p - f| / |f|`` instead of
            the absolute one (for ``1/x``, ``1/sqrt(x)``, ``2^-x`` ...).
        grid: number of candidate break points of the dynamic programme.
        samples: observed inputs of ``f``. When given, the fit minimizes the
            *expected squared error under their distribution* (data-driven
            break points); otherwise it minimizes the worst-case error over
            the whole domain.

    Returns:
        A :class:`PWL` with ``segments + 1`` knots; ``max_error`` is the
        worst-case error on ``[lo, hi]``.
    """
    if not 1 <= segments <= MAX_SEGMENTS:
        raise ValueError(f"segments must be in 1..{MAX_SEGMENTS}")
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    exact = _exact_pwl(f, lo, hi, segments)
    if exact is not None:  # ReLU, Clip, HardSigmoid, ...
        return exact
    sub = 4
    dense = np.linspace(lo, hi, (grid - 1) * sub + 1)
    fy = np.asarray(f(dense), dtype=np.float64)
    weight = (
        1.0 / np.maximum(np.abs(fy), 1e-12) if relative else np.ones_like(fy)
    )
    cand = dense[::sub]

    data_w = None
    if samples is not None and np.size(samples):
        pos = np.clip(
            np.round(
                (np.asarray(samples, dtype=np.float64).reshape(-1) - lo)
                / (hi - lo)
                * (len(dense) - 1)
            ),
            0,
            len(dense) - 1,
        ).astype(int)
        counts = np.bincount(pos, minlength=len(dense)).astype(np.float64)
        # keep every part of the domain slightly constrained
        data_w = counts / counts.sum() + 1e-3 / len(dense)

    # error between every pair of candidate knots
    err = np.full((grid, grid), np.inf)
    for i in range(grid - 1):
        j = np.arange(i + 1, grid)
        xs = dense[i * sub :]
        slope = (fy[j * sub] - fy[i * sub]) / (cand[j] - cand[i])
        line = fy[i * sub] + slope[:, None] * (xs[None, :] - cand[i])
        diff = (line - fy[i * sub :][None, :]) * weight[i * sub :][None, :]
        mask = np.arange(len(xs))[None, :] <= ((j - i) * sub)[:, None]
        if data_w is None:
            err[i, j] = np.where(mask, np.abs(diff), 0.0).max(axis=1)
        else:
            sq = (diff**2) * data_w[i * sub :][None, :]
            err[i, j] = np.where(mask, sq, 0.0).sum(axis=1)

    # dynamic programme over exactly `segments` pieces
    dp = np.full((segments + 1, grid), np.inf)
    arg = np.zeros((segments + 1, grid), dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, segments + 1):
        if data_w is None:  # minimax
            cost = np.maximum(dp[k - 1][:, None], err)
        else:  # expected squared error
            cost = dp[k - 1][:, None] + err
        arg[k] = cost.argmin(axis=0)
        dp[k] = cost.min(axis=0)
    knots_idx = [grid - 1]
    for k in range(segments, 0, -1):
        knots_idx.append(arg[k, knots_idx[-1]])
    knots = cand[np.array(knots_idx[::-1])]

    def solve(kn, iters=80):
        """Best knot values for fixed knot positions + the objective."""
        basis = _hat_basis(kn, dense)
        if data_w is None:
            vals, e = _lawson(basis, fy, weight, iters)
            return vals, basis, e
        sw = np.sqrt(data_w) * weight
        vals, *_ = np.linalg.lstsq(basis * sw[:, None], fy * sw, rcond=None)
        res = (basis @ vals - fy) * sw
        return vals, basis, float(res @ res)

    # polish the break points of the data-driven fit (deterministic LS score)
    step = (hi - lo) / (grid - 1)
    for _ in range(2 if data_w is not None else 0):
        for k in range(1, segments):
            best = solve(knots)[2]
            for cand_pos in np.linspace(knots[k] - step, knots[k] + step, 21):
                if not knots[k - 1] + 1e-9 < cand_pos < knots[k + 1] - 1e-9:
                    continue
                trial = knots.copy()
                trial[k] = cand_pos
                score = solve(trial)[2]
                if score < best:
                    best, knots = score, trial
    y, a, _ = solve(knots)
    worst = float(np.abs(weight * (a @ y - fy)).max())
    return PWL(knots, y, worst, relative)


# --------------------------------------------------- integer evaluation ----
@dataclass
class IntPWL:
    """Integer parameters of a PWL function (all arrays have ``<= 4`` rows).

    ``x_k`` / ``y_k`` are the left knots (integer codes) and ``mult`` /
    ``shift`` the slope of each segment as fixed-point multiplier. The
    evaluation is ``y_k[i] + mbqm(x - x_k[i], mult[i], shift[i])`` with
    ``i = #{ j >= 1 : x >= x_k[j] }``, then clipped to ``[qmin, qmax]``.
    """

    x_k: np.ndarray
    y_k: np.ndarray
    mult: np.ndarray
    shift: np.ndarray
    qmin: int
    qmax: int

    @property
    def segments(self):
        """Number of segments."""
        return len(self.x_k)

    def __call__(self, x):
        """Integer evaluation on an integer array (returns int32)."""
        return native.pwl(
            x, self.x_k, self.y_k, self.mult, self.shift, self.qmin, self.qmax
        )


def to_integer(pwl, in_scale, in_zp, out_scale, out_zp, qmin, qmax):
    """Maps a float :class:`PWL` onto integer codes.

    ``real = (code - zp) * scale`` on both sides. Knots that collapse onto the
    same input code are merged, so fewer than ``segments`` rows may result.
    """
    xk = np.round(pwl.x / in_scale).astype(np.int64) + int(in_zp)
    yk = np.round(pwl.y / out_scale).astype(np.int64) + int(out_zp)
    keep = [0] + [i for i in range(1, len(xk)) if xk[i] > xk[i - 1]]
    xk, yk = xk[keep], yk[keep]
    if len(xk) < 2:  # degenerate range: constant function
        xk = np.array([xk[0], xk[0] + 1])
        yk = np.array([yk[0], yk[0]])
    mult, shift = [], []
    for i in range(len(xk) - 1):
        m, s = quantize_multiplier((yk[i + 1] - yk[i]) / (xk[i + 1] - xk[i]))
        mult.append(m)
        shift.append(s)
    return IntPWL(
        xk[:-1].astype(np.int64),
        yk[:-1].astype(np.int64),
        np.array(mult, dtype=np.int64),
        np.array(shift, dtype=np.int64),
        int(qmin),
        int(qmax),
    )


def fixed_point(
    f,
    lo,
    hi,
    in_bits,
    out_bits,
    relative=True,
    segments=4,
    clip=(-(2**62), 2**62),
):
    """PWL of ``f`` on ``[lo, hi]`` as a fixed-point integer function.

    Input codes are ``round(x * 2**in_bits)``, output codes
    ``round(y * 2**out_bits)``.

    Returns:
        ``(IntPWL, PWL)``: the integer function and the float fit.
    """
    pwl = fit_pwl(f, lo, hi, segments, relative)
    ip = to_integer(pwl, 2.0**-in_bits, 0, 2.0**-out_bits, 0, *clip)
    return ip, pwl


def error_report(f, pwl, lo, hi, n=20001):
    """``(max abs error, max relative error)`` of ``pwl`` against ``f``."""
    x = np.linspace(lo, hi, n)
    ref, got = f(x), pwl(x)
    abs_err = np.abs(got - ref)
    rel = abs_err / np.maximum(np.abs(ref), 1e-12)
    return float(abs_err.max()), float(rel.max())
