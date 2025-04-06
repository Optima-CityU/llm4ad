#include <torch/extension.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int64_t* dims, int64_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int64_t dim = dims[0];
    int64_t size = dims[1];

    // Compute the reverse cumulative sum
    int64_t offset = idx;
    int64_t reverse_idx = size - offset - 1;
    float sum = 0.0f;
    for (int64_t i = reverse_idx; i < size; i++) {
        sum += input[i + offset];
        output[i + offset] = sum;
    }
}

void reverse_cumsum_forward(torch::Tensor input, torch::Tensor output, int dim) {
    int64_t num_elements = input.numel();
    int64_t dims[2] = {dim, input.size(dim)};

    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    reverse_cumsum_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dims,
        num_elements
    );
}