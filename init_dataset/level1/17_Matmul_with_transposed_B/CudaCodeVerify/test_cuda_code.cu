// matrix_multiply_kernel.cu
#include <torch/extension.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[col * K + i];  // B is transposed in memory access
        }
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Matrix multiplication with transposed B");
}