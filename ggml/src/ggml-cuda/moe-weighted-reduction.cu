#include "moe-weighted-reduction.cuh"

// The routed sum, shared by the plain and the combine kernel so both evaluate the same expression.
static __device__ __forceinline__ float moe_weighted_sum(const float * __restrict__ experts,
                                                         const float * __restrict__ expert_scale,
                                                         const float * __restrict__ weights,
                                                         const int64_t n_embd,
                                                         const int     n_expert_used,
                                                         const int64_t token,
                                                         const int64_t col) {
    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    return sum;
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    dst[token * n_embd + col] = moe_weighted_sum(experts, expert_scale, weights, n_embd, n_expert_used, token, col);
}

// dst = routed_sum + shexp * sigmoid(gate_logit), one value of gate_logit per token. The sigmoid is written as in
// unary.cu (op_sigmoid) and the mul and the add round separately (no contraction), as the unfused graph does.
// No __restrict__ on shexp/dst: dst may be the shexp buffer itself (same index, read before the write).
static __global__ void moe_weighted_combine_f32(const float * __restrict__ experts,
                                                const float * __restrict__ expert_scale,
                                                const float * __restrict__ weights,
                                                const float * __restrict__ gate_logit,
                                                const int64_t s_gate,
                                                const float * shexp,
                                                const int64_t s_shexp,
                                                float *       dst,
                                                const int64_t s_dst,
                                                const int64_t n_embd,
                                                const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    const float sum  = moe_weighted_sum(experts, expert_scale, weights, n_embd, n_expert_used, token, col);
    const float sg   = 1.0f / (1.0f + expf(-gate_logit[token * s_gate]));
    const float gated = __fmul_rn(shexp[token * s_shexp + col], sg);
    dst[token * s_dst + col] = __fadd_rn(sum, gated);
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_weighted_combine(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor *         experts,
                                       const ggml_tensor *         expert_scale,
                                       const ggml_tensor *         weights,
                                       const ggml_tensor *         gate_logit,
                                       const ggml_tensor *         shexp,
                                       ggml_tensor *               dst) {
    GGML_ASSERT(ggml_is_contiguous(experts) && ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(shexp->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];

    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_combine_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) experts->data, expert_scale ? (const float *) expert_scale->data : nullptr,
        (const float *) weights->data, (const float *) gate_logit->data, (int64_t) (gate_logit->nb[1] / sizeof(float)),
        (const float *) shexp->data, (int64_t) (shexp->nb[1] / sizeof(float)),
        (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)), n_embd, (int) n_expert_used);
    CUDA_CHECK(cudaGetLastError());
}
