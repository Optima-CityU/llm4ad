// Include necessary libraries
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[i * M + row] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose(torch::Tensor A, torch::Tensor B) {
    int M = A.size(1);  // Rows of A (after transpose)
    int K = A.size(0);  // Columns of A
    int N = B.size(1);  // Columns of B

    // Allocate output tensor C
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose", ([&] {
        matmul_transpose_kernel<scalar_t><<<grid, block>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose, "Matrix multiplication with transposed inputs");
}