// Includes required for CUDA operations
#include <torch/extension.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// Wrapper function for PyTorch binding
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get the dimensions of the input tensors
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    // Synchronize to ensure the computation is done
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA)");
}