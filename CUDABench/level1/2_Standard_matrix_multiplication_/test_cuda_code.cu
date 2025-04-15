// CUDA kernel for matrix multiplication: C = A * B

#include <torch/extension.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions of A and B
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor C
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Define block and grid size
    dim3 block(16, 16); // 16x16 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    // Synchronize to check for any errors
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication kernel (C = A * B)");
}