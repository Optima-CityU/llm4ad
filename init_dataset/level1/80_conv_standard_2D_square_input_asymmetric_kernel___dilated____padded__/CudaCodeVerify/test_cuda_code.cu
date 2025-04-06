#include <torch/extension.h>

__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int batch_size, int in_channels, int out_channels, 
    int height_in, int width_in, 
    int height_out, int width_out, 
    int kernel_height, int kernel_width, 
    int stride, int padding, 
    int dilation_h, int dilation_w)
{
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int y_out = blockIdx.z / width_out;
    int x_out = blockIdx.z % width_out;

    int y_in_start = y_out * stride - padding;
    int x_in_start = x_out * stride - padding;
    int y_in_end = y_in_start + kernel_height * dilation_h;
    int x_in_end = x_in_start + kernel_width * dilation_w;

    if (y_out < height_out && x_out < width_out) {
        float result = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int y_in = y_in_start + ky * dilation_h;
                    int x_in = x_in_start + kx * dilation_w;
                    
                    if (y_in >= 0 && y_in < height_in && x_in >= 0 && x_in < width_in) {
                        int input_idx = ((batch_idx * in_channels + c_in) * height_in + y_in) * width_in + x_in;
                        int weight_idx = ((out_channel_idx * in_channels + c_in) * kernel_height + ky) * kernel_width + kx;
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        result += bias[out_channel_idx];
        int output_idx = ((batch_idx * out_channels + out_channel_idx) * height_out + y_out) * width_out + x_out;
        output[output_idx] = result;
    }
}

void conv2d_cuda_forward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor output, 
    int stride, 
    int padding, 
    int dilation_h, 
    int dilation_w)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int height_out = output.size(2);
    const int width_out = output.size(3);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    dim3 block_dim(1, out_channels, height_out * width_out);
    dim3 grid_dim(batch_size, 1, 1);

    conv2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, 
        height_in, width_in, 
        height_out, width_out, 
        kernel_height, kernel_width, 
        stride, padding, 
        dilation_h, dilation_w
    );
}