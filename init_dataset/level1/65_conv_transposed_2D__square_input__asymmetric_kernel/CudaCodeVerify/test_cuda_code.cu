// Includes necessary header files
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 2D convolution
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels, 
    int height_in, int width_in, 
    int kernel_h, int kernel_w, 
    int stride, int padding, int output_padding, int groups) {
    
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int out_y = threadIdx.y;
    int out_x = threadIdx.x;
    
    int height_out = (height_in + 2 * padding - kernel_h + output_padding) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_w + output_padding) / stride + 1;
    
    if (out_y < height_out && out_x < width_out) {
        int input_y_start = out_y * stride - padding;
        int input_x_start = out_x * stride - padding;

        float value = 0.0f;

        for (int c_in = 0; c_in < in_channels / groups; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int input_y = input_y_start + kh;
                    int input_x = input_x_start + kw;

                    if (input_y >= 0 && input_y < height_in && input_x >= 0 && input_x < width_in) {
                        int input_idx = (batch_idx * in_channels + c_in) * height_in * width_in + input_y * width_in + input_x;
                        int weight_idx = (out_channel_idx * in_channels / groups + c_in) * kernel_h * kernel_w + kh * kernel_w + kw;

                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (bias != nullptr) {
            value += bias[out_channel_idx];
        }

        int output_idx = (batch_idx * out_channels + out_channel_idx) * height_out * width_out + out_y * width_out + out_x;
        output[output_idx] = value;
    }
}

// Wrapper for calling the CUDA kernel
void conv_transpose2d_forward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor output, 
    int stride, int padding, int output_padding, int groups) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    dim3 block(16, 16);  // Threads per block
    dim3 grid(batch_size, out_channels);  // Blocks per grid
    
    conv_transpose2d_kernel<<<grid, block>>>(
        input.data<float>(), 
        weight.data<float>(), 
        bias.data<float>(), 
        output.data<float>(), 
        batch_size, in_channels, out_channels, 
        height_in, width_in, 
        kernel_h, kernel_w, 
        stride, padding, output_padding, groups);
    
    cudaDeviceSynchronize();
}

// Expose the function to PyTorch via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward, "CUDA transposed 2D convolution");
}