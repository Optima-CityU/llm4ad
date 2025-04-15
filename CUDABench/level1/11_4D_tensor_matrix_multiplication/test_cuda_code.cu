// Include necessary headers
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 4D tensor-matrix multiplication
__global__ void tensor_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int b, int i, int j, int l, int k) {
    
    int idx_b = blockIdx.x;
    int idx_i = blockIdx.y;
    int idx_j = blockIdx.z;
    int idx_k = threadIdx.x;

    if (idx_b < b && idx_i < i && idx_j < j && idx_k < k) {
        float result = 0.0f;
        for (int l_idx = 0; l_idx < l; ++l_idx) {
            result += A[idx_b * i * j * l + idx_i * j * l + idx_j * l + l_idx] *
                     B[l_idx * k + idx_k];
        }
        C[idx_b * i * j * k + idx_i * j * k + idx_j * k + idx_k] = result;
    }
}

// Wrapper function to launch the CUDA kernel
torch::Tensor tensor_matrix_multiply_cuda(torch::Tensor A, torch::Tensor B) {
    // Get the dimensions of the input tensors
    int b = A.size(0);
    int i = A.size(1);
    int j = A.size(2);
    int l = A.size(3);
    int k = B.size(1);

    // Allocate output tensor C
    torch::Tensor C = torch::zeros({b, i, j, k}, A.options());

    // Define block and grid dimensions
    dim3 blockDim(k);
    dim3 gridDim(b, i, j);

    // Launch the kernel
    tensor_matrix_multiply_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), 
        b, i, j, l, k);

    // Return the result tensor C
    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tensor_matrix_multiply_cuda, "4D tensor-matrix multiplication");
}