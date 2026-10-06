// Native kernels for QuantAnything.
//
// Plain C ABI so the library can be loaded with ctypes (no Python headers,
// no pybind11). Every function works on contiguous row-major buffers.
//
// Integer conventions
//   * activations travel between layers as int32 arrays,
//   * "low" path : int16 operands, int32 accumulator (bits <= 15 and the
//                  accumulator bound proves no overflow),
//   * "wide" path: int32 operands, int64 accumulator.
// Requantisation is bit-exact with quantization.utils (round-half-up
// fixed-point multiply) but uses 128-bit intermediates, so it can not
// overflow for 16-bit models either.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define QA_API extern "C" __declspec(dllexport)
#else
#define QA_API extern "C" __attribute__((visibility("default")))
#endif

typedef __int128 i128;

namespace {

inline int num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int in_parallel() {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return 0;
#endif
}

// ---------------------------------------------------------------- GEMM ----
// C[M,N] = A[M,K] * B[K,N]; row-major. i-k-j order so the innermost loop is a
// contiguous axpy that the compiler vectorises.

constexpr int64_t kTileN = 512;
constexpr int64_t kTileK = 256;
constexpr int64_t kTileM = 16;

template <class TA, class TB, class TC>
void gemm_tile(int64_t i0, int64_t i1, int64_t j0, int64_t j1, int64_t N,
               int64_t K, const TA* A, const TB* B, TC* C) {
    for (int64_t i = i0; i < i1; ++i) {
        TC* c = C + i * N;
        for (int64_t j = j0; j < j1; ++j) c[j] = 0;
    }
    for (int64_t k0 = 0; k0 < K; k0 += kTileK) {
        const int64_t k1 = std::min(K, k0 + kTileK);
        for (int64_t i = i0; i < i1; ++i) {
            TC* c = C + i * N;
            const TA* a = A + i * K;
            for (int64_t k = k0; k < k1; ++k) {
                const TC av = static_cast<TC>(a[k]);
                if (av == 0) continue;
                const TB* b = B + k * N;
                for (int64_t j = j0; j < j1; ++j)
                    c[j] += av * static_cast<TC>(b[j]);
            }
        }
    }
}

template <class TA, class TB, class TC>
void gemm_serial(int64_t M, int64_t N, int64_t K, const TA* A, const TB* B,
                 TC* C) {
    for (int64_t i0 = 0; i0 < M; i0 += kTileM)
        for (int64_t j0 = 0; j0 < N; j0 += kTileN)
            gemm_tile<TA, TB, TC>(i0, std::min(M, i0 + kTileM), j0,
                                  std::min(N, j0 + kTileN), N, K, A, B, C);
}

template <class TA, class TB, class TC>
void gemm_parallel(int64_t M, int64_t N, int64_t K, const TA* A, const TB* B,
                   TC* C) {
    const int64_t tm = (M + kTileM - 1) / kTileM;
    const int64_t tn = (N + kTileN - 1) / kTileN;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (tm * tn > 1 && !in_parallel())
#endif
    for (int64_t t = 0; t < tm * tn; ++t) {
        const int64_t i0 = (t / tn) * kTileM;
        const int64_t j0 = (t % tn) * kTileN;
        gemm_tile<TA, TB, TC>(i0, std::min(M, i0 + kTileM), j0,
                              std::min(N, j0 + kTileN), N, K, A, B, C);
    }
}

// ------------------------------------------------------------- im2col ----
struct ConvShape {
    int64_t N, C, H, W, Cout, KH, KW, SH, SW, PH, PW, DH, DW, G, OH, OW;
};

// Fills cols[(c*KH*KW + ky*KW + kx), oy*OW+ox] for one image / one group.
template <class TX, class TA>
void im2col(const TX* x, const ConvShape& s, int64_t Cg, TA* cols) {
    const int64_t OHW = s.OH * s.OW;
    for (int64_t c = 0; c < Cg; ++c) {
        const TX* xc = x + c * s.H * s.W;
        for (int64_t ky = 0; ky < s.KH; ++ky) {
            for (int64_t kx = 0; kx < s.KW; ++kx) {
                TA* row = cols + ((c * s.KH + ky) * s.KW + kx) * OHW;
                for (int64_t oy = 0; oy < s.OH; ++oy) {
                    const int64_t iy = oy * s.SH - s.PH + ky * s.DH;
                    TA* dst = row + oy * s.OW;
                    if (iy < 0 || iy >= s.H) {
                        for (int64_t ox = 0; ox < s.OW; ++ox) dst[ox] = 0;
                        continue;
                    }
                    const TX* src = xc + iy * s.W;
                    for (int64_t ox = 0; ox < s.OW; ++ox) {
                        const int64_t ix = ox * s.SW - s.PW + kx * s.DW;
                        dst[ox] = (ix < 0 || ix >= s.W)
                                      ? static_cast<TA>(0)
                                      : static_cast<TA>(src[ix]);
                    }
                }
            }
        }
    }
}

// y[N, Cout, OH, OW] = conv(x[N, C, H, W], w[Cout, C/G, KH, KW])
template <class TX, class TA, class TB, class TC>
void conv_nchw(const TX* x, const TB* w, TC* y, const ConvShape& s) {
    const int64_t Cg = s.C / s.G;
    const int64_t Cog = s.Cout / s.G;
    const int64_t K = Cg * s.KH * s.KW;
    const int64_t OHW = s.OH * s.OW;
    const bool pointwise = s.KH == 1 && s.KW == 1 && s.SH == 1 && s.SW == 1 &&
                           s.PH == 0 && s.PW == 0;
    const int64_t pairs = s.N * s.G;
    const bool outer_parallel = pairs >= num_threads() && num_threads() > 1;

    auto work = [&](int64_t pair, bool inner_parallel) {
        const int64_t n = pair / s.G;
        const int64_t g = pair % s.G;
        const TX* xg = x + (n * s.C + g * Cg) * s.H * s.W;
        const TB* wg = w + g * Cog * K;
        TC* yg = y + (n * s.Cout + g * Cog) * OHW;
        std::vector<TA> buf;
        const TA* cols;
        if (pointwise) {
            buf.resize(static_cast<size_t>(K * OHW));
            for (int64_t i = 0; i < K * OHW; ++i)
                buf[i] = static_cast<TA>(xg[i]);
        } else {
            buf.resize(static_cast<size_t>(K * OHW));
            im2col<TX, TA>(xg, s, Cg, buf.data());
        }
        cols = buf.data();
        // The weights are the (Cog x K) left operand.
        if (inner_parallel)
            gemm_parallel<TB, TA, TC>(Cog, OHW, K, wg, cols, yg);
        else
            gemm_serial<TB, TA, TC>(Cog, OHW, K, wg, cols, yg);
    };

    if (outer_parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t p = 0; p < pairs; ++p) work(p, false);
    } else {
        for (int64_t p = 0; p < pairs; ++p) work(p, true);
    }
}

// ------------------------------------------------------ requantisation ----
inline int64_t clamp128(i128 v, int64_t lo, int64_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return static_cast<int64_t>(v);
}

// round(x * mult * 2^-(31+shift)), identical to utils.multiply_by_quantized_
// multiplier but with a 128-bit product.
inline i128 mbqm(int64_t x, int64_t mult, int64_t shift) {
    const i128 p = static_cast<i128>(x) * static_cast<i128>(mult);
    int64_t ts = 31 + shift;
    if (ts > 0) {
        if (ts > 120) ts = 120;
        return (p + (static_cast<i128>(1) << (ts - 1))) >> ts;
    }
    if (ts < -60) ts = -60;
    return p * (static_cast<i128>(1) << (-ts));
}

template <class TAcc>
void requantize(const TAcc* acc, int32_t* out, int64_t outer, int64_t C,
                int64_t inner, const int64_t* mult, const int64_t* shift,
                const int64_t* pre_bias, const int64_t* post_bias, int64_t zp,
                int64_t qmin, int64_t qmax) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (outer * C * inner > 65536)
#endif
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t c = 0; c < C; ++c) {
            const TAcc* a = acc + (o * C + c) * inner;
            int32_t* r = out + (o * C + c) * inner;
            const int64_t m = mult[c], sh = shift[c];
            const int64_t pb = pre_bias ? pre_bias[c] : 0;
            const int64_t qb = (post_bias ? post_bias[c] : 0) + zp;
            for (int64_t i = 0; i < inner; ++i) {
                const i128 v = mbqm(static_cast<int64_t>(a[i]) + pb, m, sh) + qb;
                r[i] = static_cast<int32_t>(clamp128(v, qmin, qmax));
            }
        }
    }
}

}  // namespace

// =============================================================== C API ====
QA_API int qa_num_threads() { return num_threads(); }

QA_API void qa_set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

QA_API int qa_has_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

// ---- GEMM ----
QA_API void qa_gemm_f32(int64_t M, int64_t N, int64_t K, const float* A,
                        const float* B, float* C) {
    gemm_parallel<float, float, float>(M, N, K, A, B, C);
}
QA_API void qa_gemm_i16_i32(int64_t M, int64_t N, int64_t K, const int16_t* A,
                            const int16_t* B, int32_t* C) {
    gemm_parallel<int16_t, int16_t, int32_t>(M, N, K, A, B, C);
}
QA_API void qa_gemm_i32_i64(int64_t M, int64_t N, int64_t K, const int32_t* A,
                            const int32_t* B, int64_t* C) {
    gemm_parallel<int32_t, int32_t, int64_t>(M, N, K, A, B, C);
}

// ---- Convolution (NCHW, weights OIHW, grouped / depthwise aware) ----
#define QA_CONV_ARGS                                                         \
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t Cout, int64_t KH,    \
        int64_t KW, int64_t SH, int64_t SW, int64_t PH, int64_t PW,          \
        int64_t DH, int64_t DW, int64_t G, int64_t OH, int64_t OW
#define QA_CONV_SHAPE \
    ConvShape { N, C, H, W, Cout, KH, KW, SH, SW, PH, PW, DH, DW, G, OH, OW }

QA_API void qa_conv_f32(const float* x, const float* w, float* y,
                        QA_CONV_ARGS) {
    conv_nchw<float, float, float, float>(x, w, y, QA_CONV_SHAPE);
}
QA_API void qa_conv_i32_i16_i32(const int32_t* x, const int16_t* w, int32_t* y,
                                QA_CONV_ARGS) {
    conv_nchw<int32_t, int16_t, int16_t, int32_t>(x, w, y, QA_CONV_SHAPE);
}
QA_API void qa_conv_i32_i32_i64(const int32_t* x, const int32_t* w, int64_t* y,
                                QA_CONV_ARGS) {
    conv_nchw<int32_t, int32_t, int32_t, int64_t>(x, w, y, QA_CONV_SHAPE);
}

// ---- Requantisation of accumulators laid out as [outer, C, inner] ----
QA_API void qa_requantize_i32(const int32_t* acc, int32_t* out, int64_t outer,
                              int64_t C, int64_t inner, const int64_t* mult,
                              const int64_t* shift, const int64_t* pre_bias,
                              const int64_t* post_bias, int64_t zp,
                              int64_t qmin, int64_t qmax) {
    requantize<int32_t>(acc, out, outer, C, inner, mult, shift, pre_bias,
                        post_bias, zp, qmin, qmax);
}
QA_API void qa_requantize_i64(const int64_t* acc, int32_t* out, int64_t outer,
                              int64_t C, int64_t inner, const int64_t* mult,
                              const int64_t* shift, const int64_t* pre_bias,
                              const int64_t* post_bias, int64_t zp,
                              int64_t qmin, int64_t qmax) {
    requantize<int64_t>(acc, out, outer, C, inner, mult, shift, pre_bias,
                        post_bias, zp, qmin, qmax);
}

// ---- Element-wise integer kernels (int32 in / int32 out) ----

// out = clip(mbqm(x - zp_in, m, s) + zp_out)
QA_API void qa_rescale(const int32_t* x, int32_t* out, int64_t n,
                       int64_t zp_in, int64_t m, int64_t s, int64_t zp_out,
                       int64_t qmin, int64_t qmax) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i)
        out[i] = static_cast<int32_t>(
            clamp128(mbqm(x[i] - zp_in, m, s) + zp_out, qmin, qmax));
}

// out = clip(mbqm(a - za, ma, sa) + sign * mbqm(b - zb, mb, sb) + zp_out)
QA_API void qa_add(const int32_t* a, const int32_t* b, int32_t* out, int64_t n,
                   int64_t za, int64_t ma, int64_t sa, int64_t zb, int64_t mb,
                   int64_t sb, int64_t sign, int64_t zp_out, int64_t qmin,
                   int64_t qmax) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i) {
        const i128 v = mbqm(a[i] - za, ma, sa) +
                       sign * mbqm(b[i] - zb, mb, sb) + zp_out;
        out[i] = static_cast<int32_t>(clamp128(v, qmin, qmax));
    }
}

// out = clip(mbqm((a - za) * (b - zb), m, s) + zp_out)
QA_API void qa_mul(const int32_t* a, const int32_t* b, int32_t* out, int64_t n,
                   int64_t za, int64_t zb, int64_t m, int64_t s, int64_t zp_out,
                   int64_t qmin, int64_t qmax) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i) {
        const int64_t prod = static_cast<int64_t>(a[i] - za) * (b[i] - zb);
        out[i] = static_cast<int32_t>(
            clamp128(mbqm(prod, m, s) + zp_out, qmin, qmax));
    }
}

// out = lut[clip(x - offset, 0, lut_len - 1)]
QA_API void qa_lut(const int32_t* x, int32_t* out, int64_t n, int64_t offset,
                   const int32_t* lut, int64_t lut_len) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i) {
        int64_t idx = static_cast<int64_t>(x[i]) - offset;
        idx = idx < 0 ? 0 : (idx >= lut_len ? lut_len - 1 : idx);
        out[i] = lut[idx];
    }
}

// ---- Float <-> integer conversion (per-tensor scale) ----

// q = clip(rint(x / scale) + zp)   (round-half-to-even, like numpy)
QA_API void qa_quantize_f32(const float* x, int32_t* q, int64_t n,
                            double scale, int64_t zp, int64_t qmin,
                            int64_t qmax) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i) {
        const double v = std::nearbyint(static_cast<double>(x[i]) / scale) +
                         static_cast<double>(zp);
        q[i] = static_cast<int32_t>(std::min<double>(
            std::max<double>(v, static_cast<double>(qmin)),
            static_cast<double>(qmax)));
    }
}

// x = (q - zp) * scale
QA_API void qa_dequantize_f32(const int32_t* q, float* x, int64_t n,
                              double scale, int64_t zp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 65536)
#endif
    for (int64_t i = 0; i < n; ++i)
        x[i] = static_cast<float>(static_cast<double>(q[i] - zp) * scale);
}
