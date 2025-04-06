// max_reduction_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_reduction_kernel(const float* input, float* output, int64_t dim_size, int64_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int base_idx = idx * dim_size;
        float max_val = -FLT_MAX;
        
        for (int i = 0; i < dim_size; ++i) {
            max_val = fmaxf(max_val, input[base_idx + i]);
        }

        output[idx] = max_val;
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor x, int dim) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor output = torch::empty({x.size(0)}, options); // assuming reduction across dim 1 for simplicity

    int64_t dim_size = x.size(1);
    int64_t num_elements = x.size(0);

    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    max_reduction_kernel<<<blocks, threads_per_block>>>(x.data_ptr<float>(), output.data_ptr<float>(), dim_size, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduction_cuda, "Max reduction CUDA kernel");
}