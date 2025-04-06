#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute masked cumulative sum along a specified dimension.
// The tensor is reshaped so that the cumulative dimension is contiguous.
template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ out,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size
) {
    // Each thread handles one "line" of the cumulative sum.
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_lines = outer_size * inner_size;
    if (index >= total_lines) return;

    // Calculate indices corresponding to the outer and inner dimensions.
    int64_t outer_idx = index / inner_size;
    int64_t inner_idx = index % inner_size;

    // Calculate the starting index for the current line.
    // The tensor is assumed to be stored in row-major order with the cumulative dimension contiguous.
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t cum_sum = 0;
    // Iterate over the cumulative dimension.
    for (int64_t d = 0; d < dim_size; d++) {
        int64_t offset = base + d * inner_size;
        // Multiply x by mask: if mask is true, use x[offset], otherwise 0.
        scalar_t value = x[offset];
        bool m = mask[offset];
        scalar_t to_add = m ? value : static_cast<scalar_t>(0);
        cum_sum += to_add;
        out[offset] = cum_sum;
    }
}

torch::Tensor masked_cumsum(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    // Ensure inputs are contiguous.
    x = x.contiguous();
    mask = mask.contiguous();

    // Get tensor dimensions and compute outer, cumulative (dim), and inner sizes.
    auto sizes = x.sizes();
    int64_t ndim = sizes.size();
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t dim_size = sizes[dim];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }

    // Prepare the output tensor.
    auto out = torch::empty_like(x);

    // Launch CUDA kernel.
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            out.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked cumulative sum (CUDA)");
}