#pragma once
#include "common.cuh"

// [PFGAP-U3] dense (non-routed) quantized prompt matmuls on bf16 tensor cores with FP32 accumulate.
// On cards whose FP32-accumulate tensor rate equals the FP16-accumulate rate, dequantizing the weight to bf16 and
// running cuBLAS beats int8 MMQ at large token counts. Detected per device when the backend starts.
// GGML_CUDA_DENSE_TC=0 off, =1 on without detection, unset = detect. GGML_CUDA_DENSE_TC_MIN = token floor (4096).
void ggml_cuda_dense_tc_detect(int device);
// true when a dense matmul of this weight type with ne11 tokens on the current device should take the bf16 path
bool ggml_cuda_dense_tc_wants(ggml_type type, int64_t ne11);
