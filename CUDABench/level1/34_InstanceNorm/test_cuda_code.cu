// instance_norm_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ inorm_weight,
    const float* __restrict__ inorm_bias,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (index < total) {
        // Determine n, c, h, w for this index
        int n = index / (C * H * W);
        int c = (index % (C * H * W)) / (H * W);
        int h = (index % (H * W)) / W;
        int w = index % W;
        
        // Compute mean and variance for the instance (n, c)
        float mean = 0.0f;
        float var = 0.0f;
        int offset = (n * C + c) * H * W;
        
        // Compute mean
        for (int i = 0; i < H * W; i++) {
            mean += x[offset + i];
        }
        mean /= (H * W);
        
        // Compute variance
        for (int i = 0; i < H * W; i++) {
            float diff = x[offset + i] - mean;
            var += diff * diff;
        }
        var /= (H * W);
        
        // Normalize the element at index
        float inv_std = rsqrtf(var + 1e-5f);
        int idx = (n * C + c) * H * W + h * W + w;
        out[idx] = (x[idx] - mean) * inv_std * inorm_weight[c] + inorm_bias[c];
    }
}

at::Tensor forward(
    const at::Tensor& x,
    const at::Tensor& inorm_weight,
    const at::Tensor& inorm_bias
) {
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(inorm_weight.dim() == 1 && inorm_weight.size(0) == x.size(1),
                "Weight tensor must be of shape (C,)");
    TORCH_CHECK(inorm_bias.dim() == 1 && inorm_bias.size(0) == x.size(1),
                "Bias tensor must be of shape (C,)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = at::empty_like(x);

    const int threads = 1024;
    const int blocks = (N * C * H * W + threads - 1) / threads;

    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        inorm_weight.data_ptr<float>(),
        inorm_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Norm CUDA forward");
}