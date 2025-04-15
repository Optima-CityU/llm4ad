// matmul_kernel.cu
#include <torch/extension.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid size
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                    (M + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    // Synchronize
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda);
}