#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

// CUDA kernel for computing C = A^T @ B^T where
// A is (K, M), B is (N, K), so A^T is (M, K), B^T is (K, N)
// Result C is (M, N)

__global__ void matmul_transpose_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[k * M + row];  // A^T[row, k] = A[k, row]
            float b_val = B[col * K + k];  // B^T[k, col] = B[col, k]
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input tensors must be 2D");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_transpose_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiply Transposed (CUDA)");
}