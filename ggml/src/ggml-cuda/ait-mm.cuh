#pragma once
// AI+ tile layout (AIT2/3/4, ggml-ait.h): CUDA mul_mat entry points (ait.cu)
#include "common.cuh"

bool ggml_cuda_ait_supported(const ggml_tensor * op);
void ggml_cuda_mul_mat_ait(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
