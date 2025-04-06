#include <torch/extension.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int height_in, 
    int width_in, 
    int height_out, 
    int width_out, 
    int kernel_size, 
    int stride, 
    int padding
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int y_out = blockIdx.z / width_out;
    int x_out = blockIdx.z % width_out;

    int y_in_start = y_out * stride - padding;
    int x_in_start = x_out * stride - padding;

    int y_in_end = min(y_in_start + kernel_size, height_in);
    int x_in_end = min(x_in_start + kernel_size, width_in);

    int y_in_start_clamped = max(y_in_start, 0);
    int x_in_start_clamped = max(x_in_start, 0);

    int output_idx = ((batch_idx * in_channels + channel_idx) * height_out + y_out) * width_out + x_out;
    
    float result = 0;
    for (int y_in = y_in_start_clamped; y_in < y_in_end; ++y_in) {
        for (int x_in = x_in_start_clamped; x_in < x_in_end; ++x_in) {
            int input_idx = ((batch_idx * in_channels + channel_idx) * height_in + y_in) * width_in + x_in;
            int weight_idx = (channel_idx * kernel_size + (y_in - y_in_start) * kernel_size + (x_in - x_in_start));
            result += input[input_idx] * weight[weight_idx];
        }
    }
    result += bias[channel_idx];
    output[output_idx] = result;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);

    int kernel_size = weight.size(2); // assuming square kernel
    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, height_out, width_out}, input.options());

    dim3 blockDim(1, in_channels, height_out * width_out);
    dim3 gridDim(batch_size, 1, 1);

    depthwise_conv2d_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        height_in, 
        width_in, 
        height_out, 
        width_out, 
        kernel_size, 
        stride, 
        padding
    );

    return output;
}