// Include necessary headers
#include <torch/extension.h>

__global__ void elu_kernel(float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = (x >= 0) ? x : alpha * (expf(x) - 1);
    }
}

torch::Tensor elu_forward(torch::Tensor input, float alpha) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    elu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, alpha);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_forward, "ELU activation forward");
}