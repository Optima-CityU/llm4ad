#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

static __inline__ __device__ void blockReduceSum(float &val, float *sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    val = sdata[0];
}

__global__ void layerNormForwardKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch_size,
    int norm_size,
    float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int offset = row * norm_size;

    float thread_sum = 0.f;
    float thread_sq_sum = 0.f;

    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float val = x[offset + i];
        thread_sum    += val;
        thread_sq_sum += val * val;
    }

    blockReduceSum(thread_sum, sdata);
    float mean = thread_sum / norm_size;

    blockReduceSum(thread_sq_sum, sdata);
    float var = thread_sq_sum / norm_size - mean * mean;
    var = var < 0.f ? 0.f : var;
    float inv_std = rsqrtf(var + eps);

    __syncthreads();
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float val  = x[offset + i];
        float norm = (val - mean) * inv_std;
        if (weight) norm *= weight[i];
        if (bias)   norm += bias[i];
        y[offset + i] = norm;
    }
}

torch::Tensor layer_norm_forward(
    torch::Tensor x,
    std::vector<int64_t> normalized_shape,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda() || !weight.defined(), "Weight must be on CUDA or undefined");
    TORCH_CHECK(bias.is_cuda()   || !bias.defined(),   "Bias must be on CUDA or undefined");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported in this example");

    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t total_size = x.numel();
    int64_t batch_size = total_size / norm_size;

    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = static_cast<int>(batch_size);
    size_t shared_mem = threads * sizeof(float);

    layerNormForwardKernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined()   && bias.numel()  > 0) ? bias.data_ptr<float>()   : nullptr,
        y.data_ptr<float>(),
        (int)batch_size,
        (int)norm_size,
        eps
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](torch::Tensor x,
           std::vector<int64_t> normalized_shape,
           torch::Tensor weight,
           torch::Tensor bias,
           float eps) {
            return layer_norm_forward(x, normalized_shape, weight, bias, eps);
        },
        "LayerNorm forward (CUDA)"
    );
}