// cuda_min_reduction.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void min_reduction_kernel(scalar_t* input, scalar_t* output, int64_t* dims, int64_t num_elements, int dim) {
    extern __shared__ scalar_t shared_data[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_size = blockDim.x;

    scalar_t min_val = 1e20;  // Set to a large value initially

    for (int i = tid; i < num_elements; i += block_size) {
        int idx = i;

        // Calculate the index based on dimension
        if (dim == 0) {
            idx = i % dims[0];
        } else {
            idx = i / dims[1];
        }

        min_val = fmin(min_val, input[idx]);
    }

    shared_data[threadIdx.x] = min_val;

    __syncthreads();

    // Perform parallel reduction
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmin(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Write result to output
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor x, int dim) {
    // Get the number of elements in the input tensor
    int64_t num_elements = x.numel();
    int64_t dims[] = {x.size(0), x.size(1)};
    torch::Tensor output = torch::empty({x.size(dim)}, x.options());

    // Launch kernel
    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "min_reduction_kernel", ([&] {
        min_reduction_kernel<scalar_t><<<num_blocks, block_size, block_size * sizeof(scalar_t)>>>(x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dims, num_elements, dim);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_reduction_cuda", &min_reduction_cuda, "Min reduction over a dimension (CUDA)");
}