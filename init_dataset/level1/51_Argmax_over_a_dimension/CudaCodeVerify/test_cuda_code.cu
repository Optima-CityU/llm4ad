// CUDA kernel to perform argmax over a specified dimension
#include <torch/extension.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, int64_t* dims, int64_t num_elements, int64_t dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        int max_index = 0;
        scalar_t max_val = input[idx * dim_size];

        for (int i = 1; i < dim_size; ++i) {
            scalar_t value = input[idx * dim_size + i];
            if (value > max_val) {
                max_val = value;
                max_index = i;
            }
        }
        output[idx] = max_index;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    // Get the shape of the input tensor
    auto sizes = input.sizes();
    int64_t num_elements = sizes[0];  // Number of elements across the batch dimension
    int64_t dim_size = sizes[dim];    // Size of the dimension along which we perform argmax

    // Flatten input tensor except the dim we're reducing over
    input = input.contiguous();
    torch::Tensor output = torch::empty({num_elements}, input.options().dtype(torch::kInt64));

    // Launch the kernel
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), output.data<int64_t>(), nullptr, num_elements, dim_size);
    }));

    return output;
}