// Required pybind11 CUDA module
#include <torch/extension.h>

// CUDA kernel for depthwise 2D convolution
__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int in_height, 
    int in_width, 
    int out_channels, 
    int out_height, 
    int out_width, 
    int kernel_h, 
    int kernel_w, 
    int stride_h, 
    int stride_w, 
    int padding_h, 
    int padding_w, 
    int dilation_h, 
    int dilation_w, 
    int groups
) {
    int batch_index = blockIdx.x;
    int channel_index = blockIdx.y;
    int out_y = threadIdx.y;
    int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    int in_channel = channel_index;
    int out_channel = channel_index;

    int in_y_start = out_y * stride_h - padding_h;
    int in_x_start = out_x * stride_w - padding_w;

    float acc = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_y = in_y_start + kh * dilation_h;
            int in_x = in_x_start + kw * dilation_w;

            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_index = (batch_index * in_channels * in_height * in_width) + 
                                  (in_channel * in_height * in_width) + 
                                  (in_y * in_width) + in_x;

                int weight_index = (out_channel * kernel_h * kernel_w) + (kh * kernel_w) + kw;

                acc += input[input_index] * weight[weight_index];
            }
        }
    }

    int output_index = (batch_index * out_channels * out_height * out_width) + 
                       (out_channel * out_height * out_width) + 
                       (out_y * out_width) + out_x;

    output[output_index] = acc + bias[out_channel];
}

// Wrapper function for PyTorch to launch the kernel
void depthwise_conv2d_forward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor output, 
    int kernel_h, 
    int kernel_w, 
    int stride_h, 
    int stride_w, 
    int padding_h, 
    int padding_w, 
    int dilation_h, 
    int dilation_w, 
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_channels = weight.size(0);
    int out_height = output.size(2);
    int out_width = output.size(3);

    dim3 block_dim(16, 16);  // Threads per block
    dim3 grid_dim(batch_size, out_channels); // Grid dimensions

    depthwise_conv2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        in_height, 
        in_width, 
        out_channels, 
        out_height, 
        out_width, 
        kernel_h, 
        kernel_w, 
        stride_h, 
        stride_w, 
        padding_h, 
        padding_w, 
        dilation_h, 
        dilation_w, 
        groups
    );
}