#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute the GELU activation
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x_val = x[idx];
        float x_cubed = x_val * x_val * x_val;
        float inner = sqrtf(2.0f / 3.14159265358979323846f) * (x_val + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);
        y[idx] = 0.5f * x_val * (1.0f + tanh_inner);
    }
}

// C++ interface function for the CUDA kernel
torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    // Ensure the input is a CUDA tensor and is contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    x = x.contiguous();
    
    const int num_elements = x.numel();
    auto y = torch::empty_like(x);

    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;
    
    // Launch the kernel
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    cudaDeviceSynchronize();  // Ensure kernel completion
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward_cuda, "GELU activation forward (CUDA)");
}