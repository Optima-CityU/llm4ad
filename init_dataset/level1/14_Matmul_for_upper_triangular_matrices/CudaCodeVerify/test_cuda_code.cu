// module_fn.cu

#include <torch/extension.h>

__global__ void matmul_upper_triangular_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && col >= row) {
        float sum = 0.0f;
        for (int k = row; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_upper_triangular(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    torch::Tensor C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    matmul_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_upper_triangular, "Upper triangular matrix multiplication");
}