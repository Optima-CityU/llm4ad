#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function to compute GELU activation (approximation)
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu(scalar_t x) {
    const scalar_t sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/pi)
    return static_cast<scalar_t>(0.5) * x * (static_cast<scalar_t>(1.0) +
           tanh(sqrt_2_over_pi * (x + static_cast<scalar_t>(0.044715) * x * x * x)));
}

// CUDA kernel for applying GELU activation element-wise
template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            const size_t numel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        const scalar_t x = input[index];
        output[index] = gelu(x);
    }
}

// Host function to launch the GELU kernel
torch::Tensor gelu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_forward_cuda", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                     output.data_ptr<scalar_t>(),
                                                     numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU activation forward (CUDA)");
}