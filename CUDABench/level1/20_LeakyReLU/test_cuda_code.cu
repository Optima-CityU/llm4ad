#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for LeakyReLU activation
template <typename scalar_t>
__global__ void leaky_relu_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  float negative_slope,
                                  size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t in_val = input[idx];
        output[idx] = (in_val > static_cast<scalar_t>(0)) ? in_val : in_val * negative_slope;
    }
}

// C++ interface that wraps the CUDA kernel
torch::Tensor leaky_relu_forward(torch::Tensor input, float negative_slope) {
    auto output = torch::empty_like(input);
    size_t num_elements = input.numel();
    
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "leaky_relu_forward", ([&] {
        leaky_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            negative_slope,
            num_elements);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}