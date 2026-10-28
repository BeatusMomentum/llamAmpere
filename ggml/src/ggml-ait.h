// AI+ tile layout (AITk, k = 2/3/4 bits per weight; ggml types ait2 / ait3 / ait4). Host reference decoder.
// The executable spec is the Python reference (AIV2/sim/aitref.py); this header, the CUDA decoder (ggml-cuda/ait.cuh)
// and the private encoder agree with it byte for byte.
//
// Tensor [K, M] (ne0 = K), M % 32 == 0, M >= 1024, K % 1024 == 0. Rotated basis per 256-K block: w' = H256 (sign . w) / 16
// with the AI+ signs (ai4_sign_bits), so y = W x = W' x' with x' rotated the same way.
// Model: w'[r][k] = lev(state) * rs[r][k / 1024] * cs[k] / 64   (rs, cs fp16; 1/64 is a format constant).
// Chunk (rt, ks) = 32 rows x 512 K (rt = 32-row tile, ks = 512-K slice, S = K / 512): AIT_CHUNK_BYTES(k) = 32 (64k + 2) =
//   [trellis 2048k bytes = 16 groups of 4 tiles] [16 fp16 row scales] [16 fp16 col scales]
//   row-scale slot i = rs[32 rt + 16 (ks % 2) + i][ks / 2];  col-scale slot i = cs[512 ks + 16 (rt % 32) + i].
//   ggml view: blck 512, type_size 64k + 2, 32-row slabs (as the unit types use 16-row slabs): chunk (rt, ks) starts at byte
//   (rt S + ks) AIT_CHUNK_BYTES(k), so rows [32 rt, 32 rt + 32) are exactly 32 nominal rows.
// Tile t (0..63) = rows 32 rt .. +31 x K cols 512 ks + 8t .. +7. Group G = t / 4, i = t % 4: tile word w (0 .. 8k-1) at
//   group byte 16 w + 4 i, group base 128 k G. Tile stream = 256k bits, bit b = bit (b % 32) of word b / 32.
// Weight n (0..255) of a tile: lane l = n / 8, j = n % 8; g = l / 4, q = l % 4; u = j / 4, h8 = (j / 2) % 2, ce = j % 2;
//   tile row 16u + g + 8h8, tile col 2q + ce (the IMMA m16n8k32 A-fragment slots). state(n) = 16 stream bits from bit n k
//   (circular), level = ((bytesum(state * tier_multiplier(k)) + 2) >> 2) - 128 (the AI+ 1MAD level).
// AIT2P (ggml type ait2p, id 113; code 1): the same bytes, chunks, tiles and scales as AIT2 (k = 2), a different level map.
//   The 256 weights of a tile form 128 pairs p = weights 2p, 2p + 1 (lane l = p / 4, j = 2 (p % 4) + ce: two adjacent
//   columns of one row). Window w_p = 20 stream bits from bit 4p (circular), entry e = bits 10..19 of (w (w + 1)) mod 2^32
//   (only the low 20 bits of w matter, so a decoder may hash any wider read), level(2p + ce) = ait2p_tab[2 e + ce]
//   (int8 1024 x 2; = AIV2/sim/tables/ait2p_tab.npy, sha256 1ca59ba8a531d2b2...).
#pragma once

#include <stdint.h>
#include <string.h>

// Format 2 uses Jake's fixed October 11 constants. Scale stretch is encoder-only.
#define AIT_FORMAT_VERSION 2u
#define AIT_FORMAT_VERSION_KEY "ait.format_version"
#define AIT_MUL_2 0xC3CCD12Du
#define AIT_MUL_3 0x87D8D12Du
#define AIT_MUL_4 0xAA73876Bu
// Type-id table: 110=AIT2, 111=AIT3, 112=AIT4; 113 uses the unchanged quadratic pair hash.
// Macro form is shared by host/device and folds to an immediate in specialized CUDA kernels.
#define AIT_MUL_FOR_TYPE(t) (((t) == 110 || (t) == 115) ? AIT_MUL_2 : ((t) == 111 || (t) == 116) ? AIT_MUL_3 : ((t) == 112 || (t) == 114 || (t) == 117) ? AIT_MUL_4 : 0u)
#define AIT_MUL_FOR_K(k) AIT_MUL_FOR_TYPE(108 + (k))
#define AIT2P_HASH_KEY "ait.pair_hash"
#define AIT2P_HASH_VERSION "quadratic20-table1024-v1"
#define AIT_TR             32
#define AIT_KS             512
#define AIT_KB             1024                      // row-scale block (K)
#define AIT_MIN_M          1024                      // col-scale slots need 32 row tiles
#define AIT_LSC            (1.0f / 64.0f)
#define AIT_TRELLIS_BYTES(k) (2048 * (k))
#define AIT_CHUNK_BYTES(k)   (32 * (64 * (k) + 2))

// the AIT2P pair table, [e][ce] (shared with the CUDA decoder, ggml-cuda/ait.cuh)
#define AIT2P_TAB_INIT { \
    4, -14, -19, 67, -47, -4, 20, -15, -37, 26, -14, 6, 13, -33, 4, -67, -6, -46, 41, -4, -40, 21, -6, -7, -5, -1, -39, 32, 92, 41, 82, 22, \
    19, -23, 34, -36, 16, 30, -19, 10, 41, 17, 1, -19, -10, -44, -36, 0, -41, -24, 11, 15, 69, -5, 3, 31, -13, -48, -31, 30, -117, 20, 47, -36, \
    47, -5, 23, 93, 2, -44, -36, 93, -19, -6, -32, 4, 31, -24, 5, -4, 43, -83, -27, -32, 60, 6, -28, 4, -73, 59, -56, -84, 21, -11, 38, -27, \
    -15, -2, -11, 66, 6, 40, 27, 14, 23, 10, -25, -3, -9, 10, -32, 26, 14, -43, -12, 16, -27, 65, -24, -11, 13, -22, 2, -16, 39, 32, -10, -6, \
    46, 103, -37, 67, -22, -1, -7, 34, -9, -8, 12, -17, -27, -22, 30, -20, 52, -104, -33, 0, 0, -6, 27, 18, -16, -92, 47, 44, -7, -36, 8, 32, \
    -31, -72, -39, -22, 69, 56, -42, -2, 19, -30, 1, 58, -51, -39, 33, -11, -22, 5, -50, 14, 13, 20, -35, -20, -17, 51, -5, -39, 11, -32, -14, 7, \
    16, 30, -15, 11, 23, -7, -63, 39, -62, 30, -21, 24, -9, 20, 46, -23, 40, 14, -14, -32, -61, -62, -19, -34, 4, -50, 11, 30, -28, 39, -5, 7, \
    25, 2, -4, -38, -52, 31, 4, 53, -2, 52, -49, -19, 0, 33, -89, 65, -3, 24, 15, 40, 52, 23, -20, 8, 38, -34, 70, 27, 10, -12, 72, -41, \
    8, -7, -34, -29, -34, 7, 22, 50, -18, 59, -16, 7, -6, -52, 38, 11, 21, -36, -32, 36, -8, -84, -27, -29, -17, -52, 34, -17, -18, -21, -57, -21, \
    -31, -35, -3, 39, -25, 72, 51, 11, -24, 14, -17, -47, -19, -25, -53, 9, 8, 16, -83, -10, -22, -20, -33, 22, -21, -74, 35, 23, -17, -29, -1, 18, \
    15, -17, 3, -56, -4, -15, 13, 2, -8, 38, 19, -8, -4, 51, -46, 13, -45, -4, -42, 79, 2, 12, 20, -44, -34, -16, -79, 20, 4, 43, -5, 82, \
    -30, -2, 25, 33, -18, -63, 1, -33, -47, 26, -66, -49, 14, -15, 48, 5, -11, -32, -22, 9, -10, 60, -19, 40, 48, 26, 39, 16, -25, -47, -33, -43, \
    -45, -18, -68, -67, 17, 54, 19, 22, -1, -51, 67, -14, -59, 80, -7, -60, -17, 31, -21, 2, 0, -2, 3, -3, 16, -3, -13, 16, -3, -56, 0, 15, \
    18, -119, -55, -11, 26, 3, 19, 79, -30, -31, 30, 61, 27, 12, -10, 19, 19, -49, 55, 7, -4, -36, 57, 63, 7, -61, -14, 32, 31, 13, 24, 9, \
    19, 32, -32, -9, 64, 24, -55, 41, 6, -16, 13, 6, 43, -48, 48, -18, -28, 19, 35, 29, -42, 10, 55, 79, -24, -33, 3, 5, 28, -29, -27, -44, \
    -7, -76, 12, -68, 31, -34, 13, -60, -38, -47, 13, -11, 59, -30, 29, -21, -35, -4, 12, -36, -14, 0, 22, 19, 20, 7, 15, -85, -4, 6, 30, -14, \
    -1, -10, -39, -7, 17, -28, 31, -67, 21, -5, -22, -14, 20, 40, 37, -17, -39, -11, 113, -50, 52, -10, 26, -18, 15, -51, 8, -37, -19, 14, -8, 3, \
    46, -47, 127, 52, -22, 86, -9, 11, -5, -9, 47, -29, -62, 15, -25, -41, 17, -26, -10, -18, -21, -17, -9, 43, -1, -17, -69, -12, -2, -5, 62, 17, \
    -60, -14, -44, -10, -33, 14, -58, -98, -1, -46, 56, 18, 41, 57, 57, -24, 41, -4, 14, 22, 9, 20, -16, 52, -46, -26, -4, -9, 12, -7, -39, 44, \
    27, 7, 7, -44, 4, -8, 33, 49, 4, -20, -37, 11, -26, -11, -4, -15, -45, -5, 5, -9, 28, 51, 13, 17, 1, 10, -6, 40, -56, 24, 17, -55, \
    1, -39, -45, -17, 56, -16, 5, -14, -40, -58, -21, -1, -15, -19, 1, 5, 33, -14, -31, -7, 25, -70, 6, 14, 77, -10, -4, 10, 0, 41, 48, 4, \
    -28, -24, 42, -29, 7, -27, 65, -3, 12, -53, -11, -29, 56, 53, -1, -29, 45, 2, 33, 17, -6, 1, -46, 39, 21, -17, 17, 17, -34, -58, -64, -25, \
    22, 9, -28, -9, -12, 13, 17, 12, 117, 5, -105, -58, 30, -25, 28, -39, -57, 21, -1, 15, 36, -30, 15, 60, -59, -7, -6, -24, -35, -124, 5, 22, \
    10, -2, -40, -75, 13, -5, -27, 34, 24, 34, 12, 26, -75, -30, -34, -40, -21, -16, -11, 37, -12, 25, 25, -14, -65, -4, -12, 7, 20, 28, 40, -42, \
    -29, -54, -30, 47, -39, 3, 23, -8, -68, -35, 5, -24, 5, 44, 7, 19, 2, 6, 9, 58, 9, -21, 48, 61, -18, -10, -21, -24, -62, 10, -9, 31, \
    -55, 1, -46, 6, 81, 31, -32, 21, 12, -13, 48, -13, 8, 1, -34, 31, -23, -5, -19, 21, 27, -35, -13, -39, 52, -23, -43, 35, 12, 43, -12, -12, \
    26, -14, -75, -39, 4, -33, 28, -2, -30, -15, 17, -68, -24, -84, 23, -53, -9, -15, 8, -50, -15, 35, 75, -20, -3, -23, -24, 4, -85, 96, 45, 36, \
    35, -6, -47, 2, 9, -6, -20, -13, 2, 26, 8, 11, 11, 54, 65, 35, 3, 84, 24, 2, -39, -3, -57, 58, -51, -68, -11, 26, -123, -20, 70, 3, \
    3, 68, -88, -90, 23, 4, 30, -60, 26, -63, 33, 27, 26, 22, -29, -2, -33, -16, 83, -61, -49, 19, 48, 1, -48, -13, 3, 7, -8, 2, -97, -14, \
    5, 8, 39, -34, -14, -5, 9, 37, 40, 3, 24, 36, 16, 5, -56, 34, 18, 67, -18, -21, 11, 48, 29, -79, -22, 19, -43, -43, -63, -17, -7, 57, \
    -18, 4, 15, 14, 45, 10, -25, -39, -22, -36, 26, -9, -29, 22, 10, -16, 2, 97, -34, -11, 21, -76, -13, -25, -85, 2, -30, 26, -19, -5, -25, -30, \
    -25, 25, -7, -21, -2, 33, 21, -102, 24, 71, -60, 1, 29, 10, 32, 34, -37, 39, -24, -17, 8, -35, -45, 4, -42, 5, -76, -6, -62, 48, 30, -43, \
    -23, -25, -86, -25, 39, 26, -44, -52, 16, -7, 46, 36, 42, 6, -16, 22, -54, 51, 32, -53, 10, -18, -52, -56, -59, 13, 42, -21, 30, 38, 38, -51, \
    -4, 30, -46, 32, -44, -21, -49, 38, -92, 17, -5, 11, -2, 3, 48, -60, -13, 9, -26, -44, 6, -5, 1, 64, 45, -54, -15, -31, -56, 8, 18, -7, \
    59, -15, 11, -16, 14, -34, 24, -12, 45, 75, 8, 27, -44, 50, -24, -22, -7, -13, 68, 73, -44, 11, -16, -15, 22, 58, 22, 23, 46, -12, 32, 42, \
    -23, 28, 49, 21, -50, -29, 32, 55, -14, -8, -47, 67, -23, 12, 39, 13, -46, 55, 64, -27, -47, 8, 7, 4, 4, -21, -43, -61, -1, -22, -28, 45, \
    7, 2, 25, -21, 30, -89, -35, 4, 6, 8, 76, 3, -16, 113, -22, -15, 26, -24, 78, 47, -45, -31, -35, 16, -12, -77, 8, 34, 19, 37, 1, -25, \
    -17, -4, -37, -18, -40, -15, 57, 2, 107, -24, 38, -7, 21, 15, -20, -12, -28, 3, 20, -18, -48, 46, -9, 14, 0, -10, 36, -22, 66, -70, -15, -8, \
    -16, 11, -12, -6, -15, -11, -13, -27, 19, -29, 45, 33, 20, -25, 43, 21, -25, 6, -29, -4, 11, 8, 29, 2, 19, 45, -29, 18, 24, -30, 19, 121, \
    -13, -2, -24, 22, -3, -27, 71, 10, -17, 3, -9, -9, 27, -9, -9, -20, -13, -27, -44, 19, 30, -48, 29, -12, 3, 48, -35, 53, 66, 45, -4, 26, \
    -53, -25, 48, 39, -29, 8, -47, 22, -22, -19, 6, -8, -23, -59, 66, -37, -42, -1, -35, -30, -66, 2, 16, 36, -26, 17, -21, 37, 33, 39, 71, -57, \
    12, 50, -25, 1, -32, -39, 31, 0, 46, -13, 7, 22, 24, 45, -49, -10, 11, 1, -47, 117, -37, 22, 35, 6, -21, 54, -8, 35, -20, -31, 24, -26, \
    9, 63, 45, 15, -14, 45, 10, 16, 7, -35, 52, -55, 47, -3, 51, -41, -2, 19, 27, -3, 18, 12, -10, -25, 24, -57, 11, -45, -2, 56, 36, 43, \
    9, 41, 32, -9, 27, -31, 43, -10, -17, -69, 85, -43, 35, -6, -32, 79, -3, -7, -5, -32, -22, 28, 28, -27, -15, 47, -19, 17, 29, -51, -67, 12, \
    3, 30, -5, -4, -7, 10, -42, 33, 43, -17, 22, 17, 17, -37, 98, 21, -42, -11, 7, 12, -31, 41, 0, -27, -10, -33, 57, -83, 24, -34, 42, -41, \
    -40, 59, -6, -17, -21, 30, -5, 16, -15, 73, -7, -110, 0, 23, -34, -14, 22, -16, -82, 32, -65, 71, 29, 30, -14, 22, 5, 4, -54, -5, -46, -34, \
    36, -21, 29, 46, -89, 38, -12, 95, 16, 4, -28, 10, 31, 21, -28, -19, 92, -98, 55, 36, -72, 16, 45, 48, 17, -41, 69, -29, -78, -15, -2, -21, \
    20, 13, 54, -5, -6, -7, -13, -18, -6, 19, 40, 22, -59, -47, 8, 2, -37, -10, 52, -36, 13, 12, -24, 39, 22, 31, 31, 8, 91, 7, 39, 51, \
    -17, 44, -32, -19, -35, 15, 18, -36, -39, -34, 39, -1, 35, 11, 3, -29, 1, -59, -28, 54, 59, 31, 20, 2, 59, -51, -4, -66, -14, -6, 18, -12, \
    21, -27, 13, 37, 2, -80, 9, -13, -27, -6, 11, 5, 6, 24, -30, 11, -17, -16, 32, -4, -29, -12, -63, -9, -6, -23, 18, 4, -17, -41, -33, -47, \
    31, 6, 1, 36, 30, 33, -37, -25, -5, 0, -21, 15, 11, 27, 54, -27, -25, -64, 4, -26, -36, -66, 10, -27, -37, -7, 23, -22, -39, -29, -78, 13, \
    -1, 22, 11, 3, 37, 8, -10, -1, -5, 46, -31, -26, -1, 27, -10, 28, 19, -15, 62, -21, 58, -1, -11, 23, -2, -10, 34, -40, -11, 48, -28, 29, \
    2, 43, -63, 24, -23, 32, 12, -31, -78, -52, -7, -7, -7, 30, -8, 48, -5, 8, -58, -26, 0, 16, 36, 88, -8, -68, 39, -69, 25, 20, 1, 24, \
    5, 17, -19, -9, 50, 17, 36, 1, -12, -37, 0, -32, 94, 70, 38, -12, 5, 2, -9, 18, -11, -63, -19, -42, 25, -48, 13, -46, -10, -53, -44, -41, \
    -7, -36, 15, 42, -10, -12, 64, 10, 21, -2, -25, -5, -14, -2, 94, -30, -25, -11, 17, -11, 51, -71, -53, -18, 35, 71, 12, 74, 56, -9, -31, 11, \
    19, -61, -8, 68, -6, -34, 5, 19, 8, -38, -18, 23, 11, -10, 57, 14, 11, -43, -4, -21, -4, 62, 26, -4, 5, -92, -36, -52, -40, 12, -99, 1, \
    -32, 20, -63, -34, 64, -45, -60, -41, 7, -75, 35, 14, 30, -11, 5, -38, -25, 48, -52, 24, -10, -35, -3, -4, -41, 18, 4, 20, 24, -26, -95, -32, \
    -6, 11, -76, 4, 35, 19, 34, -19, 21, 1, -3, 33, 84, -1, -76, 47, -43, -34, 38, -11, 6, -53, 44, -24, 94, -6, -69, 34, -9, 32, 38, 64, \
    55, 45, -6, 6, -3, 2, -1, -42, -23, 42, -15, 2, -70, -20, 13, 11, -13, 5, -50, -46, 4, 35, 9, 6, 39, 38, 2, 18, 84, 9, 26, 81, \
    54, -44, 19, 20, -12, 22, 50, 51, -9, -47, -71, 28, 17, 28, -54, -34, 78, -29, 22, -39, -9, -56, -15, 16, 82, -17, -1, 75, 15, -2, 9, -20, \
    -1, 7, -18, -55, 12, -23, 16, 25, -52, 5, -29, 13, 60, -59, -14, 40, 49, 13, 1, 1, -53, 16, 71, 18, 35, -28, 26, 26, 0, -13, -41, -41, \
    -20, -1, -39, -19, 57, -36, -86, -45, 77, 98, -6, 23, 62, -9, -35, 46, 22, 61, -23, -53, -20, 29, -25, 60, -51, -12, -17, -47, -24, -9, 16, -20, \
    -33, 61, -10, -13, 40, -15, -35, -90, -21, 47, 72, 39, 9, -12, 14, -24, -11, 1, 27, -5, 4, 11, 24, -1, -69, -7, -49, -2, -29, 1, 51, 30, \
    -1, -12, 25, 13, 10, 24, 20, 26, 18, 49, -36, 30, -105, 45, -18, -37, 38, -59, 37, -45, -26, 6, 22, -43, -22, 7, -50, -3, -16, 26, -40, 26, }

#ifdef __cplusplus
extern "C" {
#endif

static const int8_t ait2p_tab[2048] = AIT2P_TAB_INIT;

// entry of a pair window (any read whose low 20 bits are the window)
static inline int ait2p_entry(uint32_t w) {
    return (int) (((w * (w + 1u)) >> 10) & 1023u);
}

static inline float ait_h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t b;
    if (e == 0) {
        if (m == 0) { b = s; }
        else {   // subnormal
            int ee = -1; uint32_t mm = m;
            do { ++ee; mm <<= 1; } while ((mm & 0x400u) == 0);
            b = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (e == 31) { b = s | 0x7F800000u | (m << 13); }
    else { b = s | ((e + 127 - 15) << 23) | (m << 13); }
    float f; memcpy(&f, &b, 4); return f;
}

static inline int ait_level(uint32_t st, int k) {
    const uint32_t x = st * AIT_MUL_FOR_K(k);
    const int b = (int) ((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24));
    return ((b + 2) >> 2) - 128;
}

// 16 state bits of a tile at circular stream position pos
static inline uint32_t ait_state(const uint8_t * group, int i, int k, int pos) {
    const int nbits = 256 * k;
    pos %= nbits;
    const int wi = pos >> 5, sh = pos & 31, nw = 8 * k;
    uint32_t lo, hi;
    memcpy(&lo, group + 16 * wi + 4 * i, 4);
    memcpy(&hi, group + 16 * ((wi + 1) % nw) + 4 * i, 4);
    const uint32_t v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
    return v & 0xFFFFu;
}

// row scale of row m for K block m / 1024 and col scale of column k, read from the side slots
static inline float ait_row_scale(const uint8_t * base, int64_t K, int k, int64_t m, int64_t kcol) {
    const int64_t S = K / AIT_KS, rt = m / AIT_TR, rr = m % AIT_TR;
    const int64_t ks = 2 * (kcol / AIT_KB) + rr / 16;
    uint16_t h; memcpy(&h, base + (rt * S + ks) * AIT_CHUNK_BYTES(k) + AIT_TRELLIS_BYTES(k) + 2 * (rr % 16), 2);
    return ait_h2f(h);
}
static inline float ait_col_scale(const uint8_t * base, int64_t K, int k, int64_t kcol) {
    const int64_t S = K / AIT_KS, ks = kcol / AIT_KS, c = kcol % AIT_KS, rt = c / 16;
    uint16_t h; memcpy(&h, base + (rt * S + ks) * AIT_CHUNK_BYTES(k) + AIT_TRELLIS_BYTES(k) + 32 + 2 * (c % 16), 2);
    return ait_h2f(h);
}

// 32 stream bits of a k = 2 tile at circular position pos (AIT2P windows use the low 20)
static inline uint32_t ait_bits32_k2(const uint8_t * group, int i, int pos) {
    pos %= 512;
    const int wi = pos >> 5, sh = pos & 31;
    uint32_t lo, hi;
    memcpy(&lo, group + 16 * wi + 4 * i, 4);
    memcpy(&hi, group + 16 * ((wi + 1) % 16) + 4 * i, 4);
    return sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
}

// level (int) of weight (m, kcol) and its rotated-basis value; code 0 = 1MAD (AIT2/3/4), 1 = AIT2P pair table (k = 2)
static inline int ait_weight_level_c(const uint8_t * base, int64_t K, int k, int code, int64_t m, int64_t kcol) {
    const int64_t S = K / AIT_KS, rt = m / AIT_TR, ks = kcol / AIT_KS;
    const int rr = (int) (m % AIT_TR), kk = (int) (kcol % AIT_KS), t = kk / 8, c = kk % 8;
    const int u = rr / 16, h8 = (rr % 16) / 8, g = rr % 8, q = c / 2, ce = c % 2;
    const int l = 4 * g + q, j = 4 * u + 2 * h8 + ce, n = 8 * l + j;
    const uint8_t * group = base + (rt * S + ks) * AIT_CHUNK_BYTES(k) + (int64_t) (t / 4) * 128 * k;
    if (code == 1) {
        return ait2p_tab[2 * ait2p_entry(ait_bits32_k2(group, t % 4, 4 * (n / 2))) + ce];
    }
    return ait_level(ait_state(group, t % 4, k, n * k), k);
}
static inline int ait_weight_level(const uint8_t * base, int64_t K, int k, int64_t m, int64_t kcol) {
    return ait_weight_level_c(base, K, k, 0, m, kcol);
}
static inline float ait_weight_rot_c(const uint8_t * base, int64_t K, int k, int code, int64_t m, int64_t kcol) {
    return (float) ait_weight_level_c(base, K, k, code, m, kcol) * ait_row_scale(base, K, k, m, kcol) *
           ait_col_scale(base, K, k, kcol) * AIT_LSC;
}
static inline float ait_weight_rot(const uint8_t * base, int64_t K, int k, int64_t m, int64_t kcol) {
    return ait_weight_rot_c(base, K, k, 0, m, kcol);
}

#ifdef __cplusplus
}
#endif
