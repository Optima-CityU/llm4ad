// layer_norm_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(
    const float* x, const float* ln_weight, const float* ln_bias, 
    float* out, int64_t N, int64_t C, int64_t normalized_shape) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < N * C) {
        int norm_idx = index % normalized_shape;
        float sum = 0.0f;
        float square_sum = 0.0f;
        
        // Compute mean and variance for normalization
        for (int i = 0; i < normalized_shape; ++i) {
            sum += x[index];
            square_sum += x[index] * x[index];
        }

        float mean = sum / normalized_shape;
        float variance = (square_sum / normalized_shape) - (mean * mean);
        float std = sqrtf(variance + 1e-5f);
        
        // Apply layer normalization
        out[index] = (x[index] - mean) / std * ln_weight[norm_idx] + ln_bias[norm_idx];
    }
}

at::Tensor layer_norm_cuda(const at::Tensor& x, const at::Tensor& ln_weight, const at::Tensor& ln_bias, 
                           const std::vector<int64_t>& normalized_shape) {
    auto out = at::empty_like(x);
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t norm_shape = normalized_shape[0];
    
    int threads_per_block = 256;
    int blocks = (N * C + threads_per_block - 1) / threads_per_block;
    
    layer_norm_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(), ln_weight.data_ptr<float>(), ln_bias.data_ptr<float>(), 
        out.data_ptr<float>(), N, C, norm_shape
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layer_norm_cuda, "Layer normalization CUDA implementation");
}