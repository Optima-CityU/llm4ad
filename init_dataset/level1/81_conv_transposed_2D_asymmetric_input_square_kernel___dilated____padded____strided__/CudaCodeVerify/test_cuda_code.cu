// Required pybind11 CUDA module
#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int batch_size, const int in_channels, const int height_in, const int width_in,
    const int out_channels, const int height_out, const int width_out,
    const int stride, const int padding, const int dilation
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int out_row = threadIdx.y;
    int out_col = threadIdx.x;

    if (batch_idx >= batch_size || channel_idx >= out_channels || out_row >= height_out || out_col >= width_out) {
        return;
    }

    int h_start = out_row * stride - padding;
    int w_start = out_col * stride - padding;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < weight_height; ++kh) {
            for (int kw = 0; kw < weight_width; ++kw) {
                int h = h_start + kh * dilation;
                int w = w_start + kw * dilation;
                if (h >= 0 && h < height_in && w >= 0 && w < width_in) {
                    sum += x[batch_idx * in_channels * height_in * width_in + c * height_in * width_in + h * width_in + w] *
                           weight[channel_idx * in_channels * weight_height * weight_width + c * weight_height * weight_width + kh * weight_width + kw];
                }
            }
        }
    }

    output[batch_idx * out_channels * height_out * width_out + channel_idx * height_out * width_out + out_row * width_out + out_col] = sum + bias[channel_idx];
}

at::Tensor conv_transpose2d(
    at::Tensor x, 
    at::Tensor weight, 
    at::Tensor bias, 
    int stride, 
    int padding, 
    int dilation
) {
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height_in = x.size(2);
    int width_in = x.size(3);
    int out_channels = weight.size(0);
    int height_out = (height_in - 1) * stride - 2 * padding + dilation * (weight.size(2) - 1) + 1;
    int width_out = (width_in - 1) * stride - 2 * padding + dilation * (weight.size(3) - 1) + 1;

    auto output = at::zeros({batch_size, out_channels, height_out, width_out}, x.options());

    dim3 block(16, 16);
    dim3 grid(batch_size, out_channels);

    conv_transpose2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, height_in, width_in, out_channels, height_out, width_out,
        stride, padding, dilation
    );

    return output;
}