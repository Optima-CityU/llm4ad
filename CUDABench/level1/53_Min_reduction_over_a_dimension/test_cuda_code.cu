#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// A helper to convert a linear index into a multi-dimensional index.
// dims: number of dimensions in the output
// idx:  array to store the multi-dimensional index
// sizes: array of dimension sizes for the output
// linear_index: the 1D index we want to convert
__device__ inline void index_to_mdim(
    const int64_t linear_index,
    const int64_t dims,
    const int64_t* sizes,
    int64_t* idx)
{
    int64_t tmp = linear_index;
    for (int64_t d = dims - 1; d >= 0; --d)
    {
        idx[d] = tmp % sizes[d];
        tmp /= sizes[d];
    }
}

// Kernel to compute min-reduction along a single dimension.
// x          : input data pointer
// out        : output data pointer
// x_sizes    : shape of the input tensor
// x_strides  : strides of the input tensor
// out_sizes  : shape of the output tensor (input shape without the reduced dimension)
// out_strides: strides of the output tensor
// dim        : the dimension being reduced
// in_dims    : total number of dimensions in the input
// out_dims   : total number of dimensions in the output
// out_numel  : number of elements in 'out'
// reduce_size: size of the dimension being reduced
template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ x_sizes,
    const int64_t* __restrict__ x_strides,
    const int64_t* __restrict__ out_sizes,
    const int64_t* __restrict__ out_strides,
    const int64_t dim,
    const int64_t in_dims,
    const int64_t out_dims,
    const int64_t out_numel,
    const int64_t reduce_size)
{
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    // Convert the linear index into multi-dimensional index for the output
    int64_t* idx_out = new int64_t[out_dims];
    index_to_mdim(out_idx, out_dims, out_sizes, idx_out);

    // Build a full index for the input by inserting the 'reduced' dimension index later
    // Example: if in_dims=3, dim=1, then for out index [i0, i1], the full input index is [i0, "k", i1],
    // where "k" is the loop variable from 0 to reduce_size-1.
    int64_t* idx_in = new int64_t[in_dims];
    {
        // We'll fill idx_in with the same coordinates as idx_out, but skipping the reduced dimension
        int64_t out_pos = 0;
        for (int64_t d = 0; d < in_dims; d++) {
            if (d == dim) {
                // placeholder, will be filled by loop
                idx_in[d] = 0;
            } else {
                idx_in[d] = idx_out[out_pos];
                out_pos++;
            }
        }
    }

    // Perform the actual reduction (min) across the specified dimension
    scalar_t min_val = static_cast<scalar_t>(0);
    bool first_val = true;

    for (int64_t k = 0; k < reduce_size; k++) {
        idx_in[dim] = k;

        // Compute the linear offset for this particular index in 'x'
        int64_t offset = 0;
        for (int64_t d = 0; d < in_dims; d++) {
            offset += idx_in[d] * x_strides[d];
        }

        scalar_t val = x[offset];
        if (first_val) {
            min_val = val;
            first_val = false;
        } else {
            min_val = (val < min_val) ? val : min_val;
        }
    }

    // Compute the linear offset for the output index and store
    int64_t out_offset = 0;
    for (int64_t d = 0; d < out_dims; d++) {
        out_offset += idx_out[d] * out_strides[d];
    }
    out[out_offset] = min_val;

    delete[] idx_out;
    delete[] idx_in;
}

// Host function that sets up the kernel launch
torch::Tensor forward(torch::Tensor x, int64_t dim)
{
    // Ensure 'dim' is in range
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Dimension out of range");

    // Prepare output shape: same as x, but without 'dim'
    auto x_sizes = x.sizes();
    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim() - 1);
    for (int64_t d = 0; d < x.dim(); d++)
    {
        if (d == dim) continue;
        out_shape.push_back(x_sizes[d]);
    }

    // Create output tensor
    auto out = torch::empty(out_shape, x.options());

    // We will pass sizes and strides to the kernel
    // Convert them to CPU tensors of type int64_t for device transfer
    auto x_sizes_t     = torch::tensor(x_sizes, torch::dtype(torch::kInt64)).to(x.device());
    auto x_strides_t   = torch::tensor(x.strides(), torch::dtype(torch::kInt64)).to(x.device());
    auto out_sizes_t   = torch::tensor(out.sizes(), torch::dtype(torch::kInt64)).to(out.device());
    auto out_strides_t = torch::tensor(out.strides(), torch::dtype(torch::kInt64)).to(out.device());

    // Number of elements in the output
    int64_t out_numel = out.numel();
    int64_t reduce_size = x_sizes[dim];

    // Launch configuration
    const int threads = 256;
    const int blocks = (out_numel + threads - 1) / threads;

    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "min_reduce_kernel", ([&] {
        min_reduce_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            x_sizes_t.data_ptr<int64_t>(),
            x_strides_t.data_ptr<int64_t>(),
            out_sizes_t.data_ptr<int64_t>(),
            out_strides_t.data_ptr<int64_t>(),
            dim,
            x.dim(),
            out.dim(),
            out_numel,
            reduce_size
        );
    }));

    // Return the reduced tensor
    return out;
}

// PyBind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Min reduction forward (CUDA)");
}