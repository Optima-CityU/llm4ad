#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

// A small constant for numerical stability
constexpr float EPS = 1e-5f;

// Kernel 1: compute partial sums and partial squares (for mean/var) per (N, group)
__global__ void compute_sum_sum_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int N, int C, int inner_size, int groups, int group_size)
{
    // total threads handle (N * C) blocks * inner_size
    // each element belongs to exactly one group: group_index = channel / group_size
    // We'll flatten N, C, and inner_size into a single index and do partial reduction with atomic ops.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * inner_size;
    while (idx < total) {
        int inner_idx = idx % inner_size;
        int tmp = idx / inner_size;
        int c = tmp % C;
        int n = tmp / C;

        int g = c / group_size;  // group index
        float val = x[idx];
        atomicAdd(&sum[n * groups + g], val);
        atomicAdd(&sum_sq[n * groups + g], val * val);

        idx += blockDim.x * gridDim.x;
    }
}

// Kernel 2: normalize using the computed mean and variance, then apply weight/bias
__global__ void group_norm_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    int N, int C, int inner_size, int groups, int group_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * inner_size;
    while (idx < total) {
        int inner_idx = idx % inner_size;
        int tmp = idx / inner_size;
        int c = tmp % C;
        int n = tmp / C;

        int g = c / group_size;  // group index
        float mean_val = mean[n * groups + g];
        float var_val  = var[n * groups + g];
        float w = weight ? weight[c] : 1.f;
        float b = bias ? bias[c] : 0.f;

        // Normalize
        float norm_x = (x[idx] - mean_val) / sqrtf(var_val + EPS);
        // Apply weight and bias
        y[idx] = norm_x * w + b;

        idx += blockDim.x * gridDim.x;
    }
}

// Forward function: replicate F.group_norm
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups)
{
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat, "Only float32 supported for this example");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions [N, C, ...]");
    TORCH_CHECK(x.size(1) % num_groups == 0,
                "Number of channels must be divisible by num_groups");

    // Shape info
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    // Flatten the spatial dimensions into inner_size
    int64_t inner_size = 1;
    for (int d = 2; d < x.dim(); d++) {
        inner_size *= x.size(d);
    }
    int64_t group_size = C / num_groups;

    // Prepare output
    auto y = torch::empty_like(x);

    // Allocate buffers for sums and sums of squares
    auto sum    = torch::zeros({N, num_groups}, x.options());
    auto sum_sq = torch::zeros({N, num_groups}, x.options());

    // Compute partial sums
    {
        const int threads = 256;
        const int blocks = (N * C * inner_size + threads - 1) / threads;
        compute_sum_sum_sq_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            sum.data_ptr<float>(),
            sum_sq.data_ptr<float>(),
            N, C, inner_size, num_groups, group_size
        );
    }

    // Compute mean and var from sums
    // mean, var each has shape [N, num_groups]
    auto mean = sum / (group_size * inner_size);
    auto var  = sum_sq / (group_size * inner_size) - mean * mean;

    // Launch kernel to normalize and apply weight/bias
    {
        const int threads = 256;
        const int blocks = (N * C * inner_size + threads - 1) / threads;
        group_norm_forward_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.numel() > 0 ? weight.data_ptr<float>() : nullptr,
            bias.numel()   > 0 ? bias.data_ptr<float>()   : nullptr,
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            N, C, inner_size, num_groups, group_size
        );
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GroupNorm forward (CUDA)");
}