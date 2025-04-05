// softplus_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = logf(1.0f + expf(x[index]));
    }
}

torch::Tensor softplus_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    const int threads_per_block = 1024;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    softplus_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements
    );
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_forward, "Softplus activation forward");
}