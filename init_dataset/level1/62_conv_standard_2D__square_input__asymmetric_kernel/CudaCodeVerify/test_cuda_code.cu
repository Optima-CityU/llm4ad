// module_fn CUDA kernel for 2D convolution
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size, int in_channels, int height, int width, 
    int out_channels, int kernel_size, 
    int stride, int padding, int dilation, int groups,
    int height_out, int width_out) 
{
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int y_out = blockIdx.z / width_out;
    int x_out = blockIdx.z % width_out;

    int y_in = y_out * stride - padding;
    int x_in = x_out * stride - padding;
    
    float value = 0.0f;
    
    // Apply dilation and padding
    for (int c = 0; c < in_channels; c++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int y_dilated = y_in + ky * dilation;
                int x_dilated = x_in + kx * dilation;

                if (y_dilated >= 0 && y_dilated < height && x_dilated >= 0 && x_dilated < width) {
                    int input_idx = ((batch_idx * in_channels + c) * height + y_dilated) * width + x_dilated;
                    int weight_idx = ((out_channel_idx * in_channels + c) * kernel_size + ky) * kernel_size + kx;
                    value += x[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (batch_idx < batch_size && out_channel_idx < out_channels) {
        int output_idx = ((batch_idx * out_channels + out_channel_idx) * height_out + y_out) * width_out + x_out;
        output[output_idx] = value + bias[out_channel_idx];
    }
}

torch::Tensor conv2d_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding, 
    int dilation, 
    int groups) 
{
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);

    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, x.options());

    dim3 block_dim(out_channels, height_out * width_out, batch_size);
    dim3 grid_dim(out_channels, height_out * width_out, batch_size);
    
    conv2d_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height, width,
        out_channels, kernel_size, stride, padding, dilation, groups,
        height_out, width_out
    );
    
    cudaDeviceSynchronize();

    return output;
}