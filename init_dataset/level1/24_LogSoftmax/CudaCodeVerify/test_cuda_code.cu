// LogSoftmax CUDA kernel
#include <torch/extension.h>

__global__ void log_softmax_kernel(const float *input, float *output, int batch_size, int dim_size, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * dim_size) {
        int batch_index = index / dim_size;
        int dim_index = index % dim_size;

        // Compute log softmax for the specified dimension
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim_size; i++) {
            max_val = fmaxf(max_val, input[batch_index * dim_size + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < dim_size; i++) {
            sum_exp += expf(input[batch_index * dim_size + i] - max_val);
        }

        output[batch_index * dim_size + dim_index] = input[batch_index * dim_size + dim_index] - max_val - logf(sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim_size = input.size(1);

    int threads_per_block = 256;
    int blocks = (batch_size * dim_size + threads_per_block - 1) / threads_per_block;

    log_softmax_kernel<<<blocks, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim_size, dim);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda, "LogSoftmax activation (CUDA)");
}