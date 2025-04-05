// Includes required for CUDA and PyTorch bindings
#include <torch/extension.h>

// CUDA kernel for HardSigmoid activation
__global__ void hard_sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fminf(fmaxf(input[idx] * 0.2f + 0.5f, 0.0f), 1.0f);
    }
}

// Wrapper function to launch the CUDA kernel
void hard_sigmoid_cuda_forward(torch::Tensor input, torch::Tensor output) {
    const int size = input.numel();
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "hard_sigmoid_cuda_forward", ([&] {
        hard_sigmoid_kernel<<<blocks, threads_per_block>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), size);
    }));
}