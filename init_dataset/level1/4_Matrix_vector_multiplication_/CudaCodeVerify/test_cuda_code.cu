#include <torch/extension.h>

__global__ void matrix_vector_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

at::Tensor matrix_vector_mul_forward(const at::Tensor& A, const at::Tensor& B) {
    // A: (M, K), B: (K, 1) but B is contiguous so we use B[k] access.
    const auto M = A.size(0);
    const auto K = A.size(1);
    
    // Create output tensor C of shape (M, 1)
    auto C = at::zeros({M, 1}, A.options());
    
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    
    matrix_vector_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matrix_vector_mul_forward, "Matrix-vector multiplication forward (CUDA)");
}