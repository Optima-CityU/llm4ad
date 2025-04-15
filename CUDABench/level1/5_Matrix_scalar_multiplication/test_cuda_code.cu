// module_fn CUDA kernel

#include <torch/extension.h>

__global__ void matrix_scalar_multiply_kernel(float *A, float s, float *C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < M && idy < N) {
        int index = idy * M + idx;
        C[index] = A[index] * s;
    }
}

torch::Tensor module_fn_cuda(torch::Tensor A, float s) {
    auto M = A.size(0);
    auto N = A.size(1);

    auto C = torch::zeros_like(A);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matrix_scalar_multiply_kernel<<<grid, block>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), M, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda, "Matrix-scalar multiplication kernel");
}