#include <torch/extension.h>

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[i * M + row] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_transpose(torch::Tensor A, torch::Tensor B) {
    int M = A.size(1);  // A is (M, K), B is (K, N)
    int K = A.size(0);  // K
    int N = B.size(1);  // N

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matmul_transpose_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose, "Matrix multiplication with transpose (A.T @ B)");
}