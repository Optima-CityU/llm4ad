#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include <vector_types.h>

#define MAX_RANK 10

__global__ void warp_vector_max_reduce_dim_kernel(
    const float* __restrict__ in_data,
    float* __restrict__ out_data,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_sizes,
    int64_t rank,
    int64_t reduce_dim,
    int64_t reduce_size,
    int64_t out_numel,
    int64_t pre_reduce_stride)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) return;

    // Compute base index into in_data based on output coordinates.
    int n = rank - 1;
    int64_t local_out_sizes[MAX_RANK];
    #pragma unroll
    for (int i = 0; i < n; i++) {
        local_out_sizes[i] = out_sizes[i];
    }

    int64_t base_idx = 0;
    int64_t tmp = idx;
    #pragma unroll
    for (int i = n - 1; i >= 0; i--) {
        int64_t coord = tmp % local_out_sizes[i];
        int64_t dim = (i < reduce_dim) ? i : i + 1;
        base_idx += coord * __ldg(&in_strides[dim]);
        tmp /= local_out_sizes[i];
    }

    const float* in_ptr = in_data + base_idx;
    float max_val = -FLT_MAX;

    if (pre_reduce_stride == 1) {
        // Use vectorized loads with float4.
        const int vec_elems = 4; // each float4 contains 4 floats
        const int block_step = 8; // process 8 float4 loads = 32 elements per iteration
        int64_t r = 0;
        // Use 8 independent accumulators.
        float partial_max[block_step];
        #pragma unroll
        for (int i = 0; i < block_step; i++) {
            partial_max[i] = -FLT_MAX;
        }
        const float4* vec_ptr = reinterpret_cast<const float4*>(in_ptr);

        // Process in blocks of 32 elements.
        int64_t limit = reduce_size - (reduce_size % (block_step * vec_elems));
        for (; r < limit; r += block_step * vec_elems) {
            #pragma unroll
            for (int i = 0; i < block_step; i++) {
                float4 vals = vec_ptr[(r / vec_elems) + i];
                float* fvals = (float*)&vals;
                #pragma unroll
                for (int j = 0; j < vec_elems; j++) {
                    partial_max[i] = fmaxf(partial_max[i], fvals[j]);
                }
            }
        }
        // Reduce the 8 partial results.
        #pragma unroll
        for (int i = 0; i < block_step; i++) {
            max_val = fmaxf(max_val, partial_max[i]);
        }
        // Process remaining vectorized chunks.
        for (; r <= reduce_size - vec_elems; r += vec_elems) {
            float4 vals = *reinterpret_cast<const float4*>(in_ptr + r);
            float* fvals = (float*)&vals;
            #pragma unroll
            for (int j = 0; j < vec_elems; j++) {
                max_val = fmaxf(max_val, fvals[j]);
            }
        }
        // Process any residual elements.
        for (; r < reduce_size; r++) {
            max_val = fmaxf(max_val, in_ptr[r]);
        }
    } else {
        // When input is not contiguous along reduce dimension.
        #pragma unroll
        for (int64_t r = 0; r < reduce_size; ++r) {
            max_val = fmaxf(max_val, in_ptr[r * pre_reduce_stride]);
        }
    }
    out_data[idx] = max_val;
}

torch::Tensor forward(torch::Tensor input, int dim) {
    TORCH_CHECK(input.dim() > 0, "Input tensor must have at least 1 dimension");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");

    auto in_sizes = input.sizes();
    auto in_strides = input.strides();
    int64_t rank = input.dim();
    int64_t reduce_dim = dim;
    int64_t reduce_size = in_sizes[dim];

    std::vector<int64_t> out_sizes;
    out_sizes.reserve(rank - 1);
    for (int64_t d = 0; d < rank; d++) {
        if (d != reduce_dim)
            out_sizes.push_back(in_sizes[d]);
    }
    auto out = torch::empty(out_sizes, input.options());
    int64_t out_numel = out.numel();

    int64_t pre_reduce_stride = 1;
    for (int i = reduce_dim + 1; i < rank; ++i)
        pre_reduce_stride *= in_sizes[i];

    auto d_in_strides = torch::tensor(in_strides, torch::kInt64).to(input.device());
    auto d_out_sizes = torch::tensor(out_sizes, torch::kInt64).to(input.device());

    int threads = 512;
    int blocks = (out_numel + threads - 1) / threads;
    warp_vector_max_reduce_dim_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        d_in_strides.data_ptr<int64_t>(),
        d_out_sizes.data_ptr<int64_t>(),
        rank,
        reduce_dim,
        reduce_size,
        out_numel,
        pre_reduce_stride
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Max reduction over a dimension (CUDA) with warp-level vectorized max reduction");
}