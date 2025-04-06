#include <torch/extension.h>

__global__ void conv1d_kernel(
    const float* x, 
    const float* conv1d_weight, 
    const float* conv1d_bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int length, 
    int kernel_size, 
    int stride, 
    int dilation, 
    int length_out
) {
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int idx = threadIdx.x;

    if (batch_idx < batch_size && out_channel_idx < out_channels) {
        for (int i = idx; i < length_out; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < kernel_size; ++j) {
                int in_idx = i * stride - j * dilation;
                if (in_idx >= 0 && in_idx < length) {
                    for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {
                        sum += x[batch_idx * in_channels * length + in_channel_idx * length + in_idx] * 
                               conv1d_weight[out_channel_idx * in_channels * kernel_size + in_channel_idx * kernel_size + j];
                    }
                }
            }
            if (conv1d_bias != nullptr) {
                sum += conv1d_bias[out_channel_idx];
            }
            output[batch_idx * out_channels * length_out + out_channel_idx * length_out + i] = sum;
        }
    }
}

void conv1d_forward(
    at::Tensor x, 
    at::Tensor conv1d_weight, 
    at::Tensor conv1d_bias, 
    at::Tensor output, 
    int stride, 
    int dilation
) {
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int length = x.size(2);
    int out_channels = conv1d_weight.size(0);
    int kernel_size = conv1d_weight.size(2);
    int length_out = (length + stride - 1) / stride;

    dim3 block(256);
    dim3 grid(batch_size, out_channels);

    conv1d_kernel<<<grid, block>>>(
        x.data_ptr<float>(), 
        conv1d_weight.data_ptr<float>(), 
        conv1d_bias.defined() ? conv1d_bias.data_ptr<float>() : nullptr, 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        length, 
        kernel_size, 
        stride, 
        dilation, 
        length_out
    );
}