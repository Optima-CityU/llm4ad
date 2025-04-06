#include <torch/extension.h>

__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int height_in, 
    int width_in, 
    int height_out, 
    int width_out, 
    int kernel_size, 
    int stride, 
    int padding, 
    int output_padding, 
    int groups) 
{
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_h = blockIdx.z / width_out;
    int out_w = blockIdx.z % width_out;

    if (batch_idx >= batch_size || out_ch >= out_channels || out_h >= height_out || out_w >= width_out)
        return;

    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;

    float value = 0.0f;
    for (int c = 0; c < in_channels; c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;
                if (in_h >= 0 && in_w >= 0 && in_h < height_in && in_w < width_in) {
                    int weight_idx = (out_ch * in_channels * kernel_size * kernel_size) + 
                                      (c * kernel_size * kernel_size) + 
                                      (kh * kernel_size) + kw;
                    int input_idx = (batch_idx * in_channels * height_in * width_in) + 
                                     (c * height_in * width_in) + 
                                     (in_h * width_in) + in_w;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = (batch_idx * out_channels * height_out * width_out) + 
                     (out_ch * height_out * width_out) + 
                     (out_h * width_out) + out_w;

    if (bias != nullptr) {
        value += bias[out_ch];
    }

    output[output_idx] = value;
}

torch::Tensor transposed_conv2d(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding, 
    int output_padding, 
    int groups) 
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto height_out = (height_in - 1) * stride + kernel_size - 2 * padding + output_padding;
    auto width_out = (width_in - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 block_dim(32, 32, 1);
    dim3 grid_dim(batch_size, out_channels, height_out * width_out);
    
    transposed_conv2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        height_in, 
        width_in, 
        height_out, 
        width_out, 
        kernel_size, 
        stride, 
        padding, 
        output_padding, 
        groups
    );

    return output;
}