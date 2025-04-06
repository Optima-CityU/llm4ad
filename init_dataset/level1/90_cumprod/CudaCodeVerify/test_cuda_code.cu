#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel(const scalar_t* __restrict__ in,
                               scalar_t* __restrict__ out,
                               int64_t outer_size,
                               int64_t dim_size,
                               int64_t inner_size) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (tid >= total) return;

    int64_t outer_idx = tid / inner_size;
    int64_t inner_idx = tid % inner_size;
    int64_t base = (outer_idx * dim_size) * inner_size + inner_idx;

    scalar_t val = in[base];
    out[base] = val;
    for (int64_t i = 1; i < dim_size; ++i) {
        int64_t idx = base + i * inner_size;
        val = val * in[idx];
        out[idx] = val;
    }
}

at::Tensor cumprod_cuda(at::Tensor x, int64_t dim) {
    auto input = x.contiguous();
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;
    auto sizes = input.sizes();

    int64_t dim_size = sizes[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= sizes[i];
    }

    auto output = at::empty_like(input);
    int64_t total = outer_size * inner_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size, dim_size, inner_size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda, "Cumulative product along a dimension (CUDA)");
}