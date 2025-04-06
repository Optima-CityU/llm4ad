// cuda_transpose_conv2d.cu

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float *x, 
    const float *weight, 
    const float *bias, 
    float *output, 
    const int batch_size, 
    const int in_channels, 
    const int out_channels, 
    const int height_in, 
    const int width_in, 
    const int height_out, 
    const int width_out, 
    const int stride, 
    const int padding, 
    const int output_padding, 
    const int groups) 
{
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int y_out = blockIdx.z / width_out;
    int x_out = blockIdx.z % width_out;

    int in_channel_start = out_channel * in_channels / groups;
    int in_channel_end = (out_channel + 1) * in_channels / groups;

    if (batch_idx < batch_size && out_channel < out_channels && y_out < height_out && x_out < width_out) {
        float sum = 0.0f;

        // Iterate over the input channels for grouped convolutions
        for (int in_channel = in_channel_start; in_channel < in_channel_end; ++in_channel) {
            for (int y_in = max(0, y_out * stride - padding); y_in < min(height_in, y_out * stride - padding + weight[2]); ++y_in) {
                for (int x_in = max(0, x_out * stride - padding); x_in < min(width_in, x_out * stride - padding + weight[3]); ++x_in) {
                    int weight_idx = (((out_channel * in_channels + in_channel) * weight[2] + (y_out * stride - y_in)) * weight[3] + (x_out * stride - x_in));
                    int input_idx = ((batch_idx * in_channels + in_channel) * height_in + y_in) * width_in + x_in;
                    sum += x[input_idx] * weight[weight_idx];
                }
            }
        }

        sum += bias[out_channel]; // Adding bias
        output[((batch_idx * out_channels + out_channel) * height_out + y_out) * width_out + x_out] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int groups) {
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int height_in = x.size(2);
    const int width_in = x.size(3);

    const int out_channels = weight.size(0);
    const int height_out = (height_in - 1) * stride - 2 * padding + weight.size(2) + output_padding;
    const int width_out = (width_in - 1) * stride - 2 * padding + weight.size(3) + output_padding;

    torch::Tensor output = torch::zeros({batch_size, out_channels, height_out, width_out}, x.options());

    const int threads_per_block = 256;
    const dim3 threads(threads_per_block);
    const dim3 blocks(batch_size, out_channels, height_out * width_out);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
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
        stride,
        padding,
        output_padding,
        groups
    );

    return output;
}