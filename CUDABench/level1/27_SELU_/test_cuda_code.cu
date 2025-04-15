// selu_kernel.cu
#include <torch/extension.h>
#include <math.h>

__device__ float selu_single(float x) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    return (x > 0) ? scale * x : scale * alpha * (expf(x) - 1);
}

__global__ void selu_kernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = selu_single(input[idx]);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    selu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_cuda, "SELU activation");
}