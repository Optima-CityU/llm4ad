#include <torch/extension.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int height_in, int width_in,
    int out_channels, int height_out, int width_out,
    int kernel_height, int kernel_width, 
    int stride, int padding, int dilation, int groups
) {
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int out_y = blockIdx.z / width_out;
    int out_x = blockIdx.z % width_out;

    int in_channel_start = out_channel_idx * in_channels / groups;
    int in_channel_end = in_channel_start + in_channels / groups;

    float result = 0.0f;

    for (int c = in_channel_start; c < in_channel_end; ++c) {
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < height_in && in_x >= 0 && in_x < width_in) {
                    int input_idx = ((batch_idx * in_channels + c) * height_in + in_y) * width_in + in_x;
                    int weight_idx = ((out_channel_idx * in_channels + c) * kernel_height + ky) * kernel_width + kx;
                    result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        result += bias[out_channel_idx];
    }

    int output_idx = ((batch_idx * out_channels + out_channel_idx) * height_out + out_y) * width_out + out_x;
    output[output_idx] = result;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, int groups
) {
    auto output = torch::zeros({input.size(0), weight.size(0), 
                                (input.size(2) + 2 * padding - weight.size(2)) / stride + 1,
                                (input.size(3) + 2 * padding - weight.size(3)) / stride + 1}, 
                                input.options());

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int out_channels = weight.size(0);
    int height_out = output.size(2);
    int width_out = output.size(3);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks(batch_size, out_channels, height_out * width_out);

    conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, height_in, width_in,
        out_channels, height_out, width_out, kernel_height, kernel_width, 
        stride, padding, dilation, groups
    );

    return output;
}