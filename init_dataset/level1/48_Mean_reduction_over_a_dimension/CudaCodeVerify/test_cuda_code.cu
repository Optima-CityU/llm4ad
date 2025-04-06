// module_fn_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduce_kernel(const float* input, float* output, int64_t* input_strides, int64_t* output_strides, int64_t dim_size, int64_t total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    int reduced_idx = idx;
    int block_size = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        sum += input[reduced_idx];
        reduced_idx += input_strides[1];  // Adjust for the dimension stride
    }

    output[idx] = sum / dim_size;
}

torch::Tensor mean_reduce(torch::Tensor x, int dim) {
    auto input = x.contiguous();
    auto output = torch::empty_like(input);

    int64_t total_elements = input.numel() / input.size(dim);
    int64_t dim_size = input.size(dim);

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    mean_reduce_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.strides().data(),
        output.strides().data(),
        dim_size,
        total_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_reduce", &mean_reduce, "CUDA mean reduce operation");
}