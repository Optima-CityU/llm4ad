#include <torch/extension.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;

    if (n < N && m < M && l < L) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[n * M * L + m * L + l] = value;
    }
}

at::Tensor matmul_cuda(at::Tensor A, at::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, (L + block.z - 1) / block.z);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "3D tensor-matrix multiplication (CUDA)");
}