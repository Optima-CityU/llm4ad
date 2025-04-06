// Include necessary headers
#include <torch/extension.h>

__global__ void l1_normalize_kernel(float* x, float* out, int64_t* dims, int64_t N, int64_t D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Calculate sum of absolute values along dimension D
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            sum += fabsf(x[idx * D + d]);
        }

        // Normalize by the L1 norm
        for (int d = 0; d < D; ++d) {
            out[idx * D + d] = x[idx * D + d] / sum;
        }
    }
}

void l1_normalize_forward(torch::Tensor x, torch::Tensor out) {
    auto N = x.size(0);  // Assuming the first dimension corresponds to N
    auto D = x.size(1);  // Assuming the second dimension corresponds to D

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    l1_normalize_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), nullptr, N, D);
    cudaDeviceSynchronize();
}