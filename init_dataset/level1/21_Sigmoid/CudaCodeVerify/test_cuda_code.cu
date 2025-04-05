#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to apply the sigmoid activation function element-wise.
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               int64_t numel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        // Compute the sigmoid function: 1 / (1 + exp(-x))
        output[index] = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp(-input[index]));
    }
}

// C++ interface to launch the CUDA kernel.
torch::Tensor sigmoid_cuda(torch::Tensor input) {
    // Ensure the input is contiguous.
    input = input.contiguous();
    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    
    // Define CUDA kernel launch configuration.
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_cuda", ([&] {
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
    }));
    
    return output;
}

// Pybind11 module definition.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sigmoid_cuda, "Apply Sigmoid activation (CUDA)");
}