// Batch normalization CUDA kernel
#include <torch/extension.h>

__global__ void batch_norm_kernel(
    const float* x, const float* bn_weight, const float* bn_bias,
    const float* bn_running_mean, const float* bn_running_var,
    float* output, int batch_size, int num_features, int spatial_dim,
    float eps, bool training) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * num_features * spatial_dim) {
        int batch_idx = idx / (num_features * spatial_dim);
        int feature_idx = (idx / spatial_dim) % num_features;
        int spatial_idx = idx % spatial_dim;

        // Get the running mean and variance for the current feature
        float mean = bn_running_mean[feature_idx];
        float var = bn_running_var[feature_idx];

        // If in training mode, compute the normalized value
        float x_val = x[idx];
        float normalized_val = (x_val - mean) / sqrtf(var + eps);

        // Apply scale and shift (gamma and beta)
        float output_val = bn_weight[feature_idx] * normalized_val + bn_bias[feature_idx];

        // Store the result
        output[idx] = output_val;
    }
}

void batch_norm_forward(
    torch::Tensor x, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_running_mean, torch::Tensor bn_running_var,
    torch::Tensor output, float eps, bool training) {

    int batch_size = x.size(0);
    int num_features = x.size(1);
    int spatial_dim = x.numel() / (batch_size * num_features);

    int block_size = 256;
    int num_blocks = (batch_size * num_features * spatial_dim + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(), bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_features, spatial_dim,
        eps, training);

    cudaDeviceSynchronize();
}