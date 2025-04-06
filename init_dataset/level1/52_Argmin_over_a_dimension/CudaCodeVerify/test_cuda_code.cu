#include <torch/extension.h>

__global__ void argmin_kernel(const float* x, int* indices, int num_elements, int dim, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int min_idx = idx;
        float min_val = x[idx * stride];
        for (int i = 1; i < stride; ++i) {
            int current_idx = idx * stride + i;
            if (x[current_idx] < min_val) {
                min_val = x[current_idx];
                min_idx = current_idx;
            }
        }
        indices[idx] = min_idx % stride;
    }
}

torch::Tensor argmin_cuda(torch::Tensor x, int dim) {
    // Set up kernel launch configuration
    const int total_elements = x.size(dim);
    const int num_elements = x.numel() / total_elements;
    
    torch::Tensor indices = torch::empty({num_elements}, torch::kInt32);
    
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    argmin_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(), 
        indices.data_ptr<int>(), 
        num_elements, 
        dim, 
        total_elements
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda, "CUDA-based argmin operation");
}