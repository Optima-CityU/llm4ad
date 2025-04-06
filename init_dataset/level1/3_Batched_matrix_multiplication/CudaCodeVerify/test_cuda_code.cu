#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

// CUDA kernel for batched matrix multiplication
__global__ void batchedMatMulKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int m, int n, int k) {
    // Compute indices for output element (row, col) and batch index.
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int batch = blockIdx.z;

    if (row < m && col < n) {
        float sum = 0.0f;
        // Pointers to the start of the batch for A and B
        const float* A_batch = A + batch * m * k;
        const float* B_batch = B + batch * k * n;
        for (int i = 0; i < k; ++i) {
            sum += A_batch[row * k + i] * B_batch[i * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

void batchedMatMulLauncher(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto k = A.size(2);
    const auto n = B.size(2);

    // Define block and grid dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size);

    // Launch kernel
    batchedMatMulKernel<<<grid, block>>>(A.data_ptr<float>(),
                                          B.data_ptr<float>(),
                                          C.data_ptr<float>(),
                                          m, n, k);

    // Check for errors in kernel launch (optional)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    // Check inputs are CUDA tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Input tensors must be CUDA tensors");
    }

    // Get dimensions
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto k = A.size(2);
    const auto n = B.size(2);

    // Allocate output tensor
    auto C = at::empty({batch_size, m, n}, A.options());

    // Launch CUDA kernel
    batchedMatMulLauncher(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Batched Matrix Multiplication (CUDA)");
}