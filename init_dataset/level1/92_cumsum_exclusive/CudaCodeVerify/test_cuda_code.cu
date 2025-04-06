// C++/CUDA code using pybind11 to implement the exclusive cumulative sum

#include <torch/extension.h>

__global__ void exclusive_cumsum_kernel(const float* input, float* output, int64_t* dims, int64_t dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dim_size) {
        // Find exclusive cumulative sum along the specified dimension
        int64_t idx_before = (idx == 0) ? 0 : input[idx - 1];
        output[idx] = idx_before + input[idx];
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim) {
    auto output = torch::zeros_like(input);
    int64_t dim_size = input.size(dim);
    
    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (dim_size + threads_per_block - 1) / threads_per_block;
    
    exclusive_cumsum_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.sizes().data(),
        dim_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exclusive_cumsum", &exclusive_cumsum_cuda, "Exclusive Cumulative Sum (CUDA)");
}