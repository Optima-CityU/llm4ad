#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

static __forceinline__ __device__ 
void index_to_coord_excluding_dim(
    int64_t idx,
    const int64_t* out_sizes,    // shape without the reduced dim
    int64_t out_ndims,          // ndims = original_ndims - 1
    int64_t* out_coord)
{
    for (int64_t d = out_ndims - 1; d >= 0; --d) {
        out_coord[d] = idx % out_sizes[d];
        idx /= out_sizes[d];
    }
}

// Each thread processes one output element (which corresponds
// to a unique combination of all dims except the reduced dim).
template <typename scalar_t>
__global__ void argmax_kernel(
    const scalar_t* __restrict__ input,
    int64_t* __restrict__ output,
    const int64_t* __restrict__ in_sizes,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_sizes,   // same as in_sizes but with dimension 'dim' removed
    const int64_t* __restrict__ out_strides, // strides for the output shape
    const int64_t dim_size,                  // in_sizes[dim]
    const int64_t out_elems,                 // product of out_sizes
    const int64_t in_ndims,                  // original number of dims
    const int64_t out_ndims,                 // in_ndims - 1
    const int64_t dim                        // dimension to reduce
)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_elems) {
        return;
    }

    // Coordinate in the output (ndims-1)
    // We'll build a fullCoord for input by inserting the reduced index.
    int64_t out_coord[16];     // supports up to 16D
    int64_t full_coord[16];    // coordinate in the original input

    // Convert 'idx' into coordinates for the output shape
    index_to_coord_excluding_dim(idx, out_sizes, out_ndims, out_coord);

    // Build the full_coord by inserting 0 in 'dim' and mapping other indices
    {
        int64_t od = 0;
        for (int64_t d = 0; d < in_ndims; d++) {
            if (d == dim) {
                full_coord[d] = 0; // placeholder
            } else {
                full_coord[d] = out_coord[od++];
            }
        }
    }

    // We'll find the index of the max value along dimension 'dim'
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int64_t max_idx = 0;

    // Initial offset for the coordinate with full_coord[dim] = 0
    int64_t base_offset = 0;
    for (int64_t d = 0; d < in_ndims; d++) {
        base_offset += full_coord[d] * in_strides[d];
    }

    for (int64_t k = 0; k < dim_size; k++) {
        // offset for input (increment along 'dim')
        int64_t offset = base_offset + k * in_strides[dim];
        scalar_t val = input[offset];
        if (val > max_val) {
            max_val = val;
            max_idx = k;
        }
    }

    // Write the argmax index
    // We need to find output offset from out_coord
    int64_t out_offset = 0;
    for (int64_t d = 0; d < out_ndims; d++) {
        out_offset += out_coord[d] * out_strides[d];
    }
    output[out_offset] = max_idx;
}

torch::Tensor argmax_forward(torch::Tensor input, int64_t dim)
{
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");

    // Make sure input is contiguous
    auto x = input.contiguous();
    auto sizes = x.sizes();
    auto strides = x.strides();
    const int64_t ndims = x.dim();

    // Prepare output shape: same as input but remove dimension 'dim'
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndims - 1);
    for (int64_t d = 0; d < ndims; d++) {
        if (d == dim) continue;
        out_shape.push_back(sizes[d]);
    }

    auto out = torch::empty(out_shape, x.options().dtype(torch::kLong));
    int64_t out_elems = 1;
    for (auto s : out_shape) {
        out_elems *= s;
    }

    // Copy sizes/strides to device
    std::vector<int64_t> h_in_sizes(ndims), h_in_strides(ndims);
    for (int64_t i = 0; i < ndims; i++) {
        h_in_sizes[i] = sizes[i];
        h_in_strides[i] = strides[i];
    }

    std::vector<int64_t> h_out_sizes(ndims - 1), h_out_strides(ndims - 1);
    auto out_sizes = out.sizes();
    auto out_strides = out.strides();
    for (int64_t i = 0; i < ndims - 1; i++) {
        h_out_sizes[i]   = out_sizes[i];
        h_out_strides[i] = out_strides[i];
    }

    int64_t* d_in_sizes;
    int64_t* d_in_strides;
    int64_t* d_out_sizes;
    int64_t* d_out_strides;
    size_t sz_in  = ndims       * sizeof(int64_t);
    size_t sz_out = (ndims - 1) * sizeof(int64_t);

    cudaMalloc(&d_in_sizes,   sz_in);
    cudaMalloc(&d_in_strides, sz_in);
    cudaMalloc(&d_out_sizes,   sz_out);
    cudaMalloc(&d_out_strides, sz_out);

    cudaMemcpy(d_in_sizes,   h_in_sizes.data(),   sz_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, h_in_strides.data(), sz_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_sizes,  h_out_sizes.data(),  sz_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides,h_out_strides.data(),sz_out, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (out_elems + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "argmax_cuda_forward", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<int64_t>(),
            d_in_sizes,
            d_in_strides,
            d_out_sizes,
            d_out_strides,
            sizes[dim],
            out_elems,
            ndims,
            ndims - 1,
            dim
        );
    }));

    cudaFree(d_in_sizes);
    cudaFree(d_in_strides);
    cudaFree(d_out_sizes);
    cudaFree(d_out_strides);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward, "Argmax forward (CUDA)");
}