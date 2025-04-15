#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for Softsign activation
__global__ void softsign_kernel(const float* __restrict__ x, float* __restrict__ y, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float val = x[index];
        y[index] = val / (1.0f + fabsf(val));
    }
}

// C++ interface function to launch the CUDA kernel
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    auto y = torch::empty_like(x);
    int size = x.numel();
    
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    
    // Launch the kernel
    softsign_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation forward (CUDA)");
}