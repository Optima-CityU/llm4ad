#include <torch/extension.h>

__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor tril_matmul_forward(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    tril_matmul_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tril_matmul_forward, "Matrix multiplication of lower triangular matrices A and B.");
}