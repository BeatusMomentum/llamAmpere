#pragma once
// AI+ tile layout (AITk, k = 2/3/4): device decoder shared by the ggml CUDA backend and the encoder.
// Format: ggml/src/ggml-ait.h. Header-only so both sides compile the same decode.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "../ggml-ait.h"


// AIT2P pair table [e][ce] (ggml-ait.h); the fast kernels copy it into shared memory
static __device__ const int8_t ait2p_dev_tab[2048] = AIT2P_TAB_INIT;

static __device__ __forceinline__ int ait2p_dev_entry(const uint32_t w) {
    return (int) (((w * w + w) >> 10) & 1023u);
}

static __device__ __forceinline__ int ait_dev_level(const uint32_t st, const int k) {
    const uint32_t x = st * AIT_MUL_FOR_K(k);
    return (int) ((__dp4a(x, 0x01010101u, 2u)) >> 2) - 128;
}

static __device__ __forceinline__ int64_t ait_dev_chunk_bytes(const int k) { return 32 * (64 * k + 2); }

// 16 state bits at circular stream position pos of tile i of a group
static __device__ __forceinline__ uint32_t ait_dev_state(const uint8_t * __restrict__ group, const int i, const int k, int pos) {
    const int nbits = 256 * k, nw = 8 * k;
    pos = pos >= nbits ? pos - nbits : pos;
    const int wi = pos >> 5, sh = pos & 31;
    const int wj = wi + 1 == nw ? 0 : wi + 1;
    const uint32_t lo = *reinterpret_cast<const uint32_t *>(group + 16 * wi + 4 * i);
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(group + 16 * wj + 4 * i);
    return (sh ? __funnelshift_r(lo, hi, sh) : lo) & 0xFFFFu;
}

// 32 stream bits at circular position pos of tile i of a k = 2 group (AIT2P windows: the low 20)
static __device__ __forceinline__ uint32_t ait_dev_bits32_k2(const uint8_t * __restrict__ group, const int i, int pos) {
    pos &= 511;
    const int wi = pos >> 5, sh = pos & 31, wj = (wi + 1) & 15;
    const uint32_t lo = *reinterpret_cast<const uint32_t *>(group + 16 * wi + 4 * i);
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(group + 16 * wj + 4 * i);
    return sh ? __funnelshift_r(lo, hi, sh) : lo;
}

// level of weight (rr, kk) inside chunk c (rr < 32 tile row, kk < 512 slice column); code 1 = AIT2P (k = 2)
static __device__ __forceinline__ int ait_dev_weight_level(const uint8_t * __restrict__ chunk, const int k, const int rr, const int kk,
                                                           const int code = 0) {
    const int t = kk >> 3, c = kk & 7;
    const int u = rr >> 4, h8 = (rr >> 3) & 1, g = rr & 7, q = c >> 1, ce = c & 1;
    const int n = 8 * (4 * g + q) + 4 * u + 2 * h8 + ce;
    if (code == 1) {
        return ait2p_dev_tab[2 * ait2p_dev_entry(ait_dev_bits32_k2(chunk + (t >> 2) * 256, t & 3, 4 * (n >> 1))) + ce];
    }
    return ait_dev_level(ait_dev_state(chunk + (t >> 2) * 128 * k, t & 3, k, n * k), k);
}

static __device__ __forceinline__ float ait_dev_row_scale(const uint8_t * __restrict__ base, const int64_t K, const int k,
                                                          const int64_t m, const int64_t kcol) {
    const int64_t S = K / 512, rt = m / 32, rr = m % 32, ks = 2 * (kcol / 1024) + rr / 16;
    const __half h = *reinterpret_cast<const __half *>(base + (rt * S + ks) * ait_dev_chunk_bytes(k) + 2048 * k + 2 * (rr % 16));
    return __half2float(h);
}

static __device__ __forceinline__ float ait_dev_col_scale(const uint8_t * __restrict__ base, const int64_t K, const int k,
                                                          const int64_t kcol) {
    const int64_t S = K / 512, ks = kcol / 512, c = kcol % 512, rt = c / 16;
    const __half h = *reinterpret_cast<const __half *>(base + (rt * S + ks) * ait_dev_chunk_bytes(k) + 2048 * k + 32 + 2 * (c % 16));
    return __half2float(h);
}
