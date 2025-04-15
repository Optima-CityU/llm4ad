// Swish activation kernel
#include <torch/extension.h>

__global__ void swish_kernel(const float* x, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float sigmoid_val = 1.0f / (1.0f + expf(-x[idx]));
        out[idx] = x[idx] * sigmoid_val;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads_per_block = 1024;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    swish_kernel<<<blocks, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward");
}