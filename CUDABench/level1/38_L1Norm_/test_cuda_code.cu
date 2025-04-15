#include <torch/extension.h>

__global__ void l1_normalize_kernel(float* input, float* output, int64_t batch_size, int64_t dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int64_t b = idx / dim;
        int64_t d = idx % dim;
        // Calculate sum of absolute values along the dimension
        float sum = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            sum += fabsf(input[b * dim + i]);
        }
        // Normalize by the sum of absolute values
        output[idx] = input[idx] / sum;
    }
}

at::Tensor l1_normalize_forward(at::Tensor input) {
    auto output = at::empty_like(input);
    int64_t batch_size = input.size(0);
    int64_t dim = input.size(1);
    
    int threads = 256;
    int blocks = (batch_size * dim + threads - 1) / threads;
    
    l1_normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &l1_normalize_forward, "L1 normalization kernel");
}