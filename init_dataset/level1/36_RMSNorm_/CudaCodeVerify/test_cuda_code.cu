#include <torch/extension.h>

__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out, const int N, const int C, const int H, const int W, const float eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < N * H * W) {
        int batch_idx = index / (H * W);
        int spatial_idx = index % (H * W);
        
        float sum_sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            int idx = batch_idx * C * H * W + c * H * W + spatial_idx;
            sum_sq += x[idx] * x[idx];
        }

        float rms = sqrtf(sum_sq / C + eps);
        for (int c = 0; c < C; ++c) {
            int idx = batch_idx * C * H * W + c * H * W + spatial_idx;
            out[idx] = x[idx] / rms;
        }
    }
}

torch::Tensor rms_norm_forward(torch::Tensor x, float eps) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto out = torch::empty_like(x, options);

    const int N = x.size(0); // batch size
    const int C = x.size(1); // number of channels
    const int H = x.size(2); // height
    const int W = x.size(3); // width

    const int threads = 1024;
    const int blocks = (N * H * W + threads - 1) / threads;

    rms_norm_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, eps);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_forward, "RMS normalization forward CUDA kernel");
}