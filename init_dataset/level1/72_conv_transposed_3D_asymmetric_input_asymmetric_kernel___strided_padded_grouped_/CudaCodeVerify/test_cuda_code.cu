// cuda_transpose3d_conv_kernel.cu

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float *input, 
    const float *weight, 
    const float *bias, 
    float *output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth_in, 
    int height_in, 
    int width_in, 
    int depth_out, 
    int height_out, 
    int width_out, 
    int kernel_depth, 
    int kernel_height, 
    int kernel_width, 
    int stride_depth, 
    int stride_height, 
    int stride_width, 
    int padding_depth, 
    int padding_height, 
    int padding_width, 
    int output_padding_depth, 
    int output_padding_height, 
    int output_padding_width, 
    int groups) 
{
    // Get the current thread's batch, output depth, height, and width
    int b = blockIdx.x;
    int d_out = blockIdx.y;
    int h_out = blockIdx.z;
    int w_out = threadIdx.x;

    if (b >= batch_size || d_out >= depth_out || h_out >= height_out || w_out >= width_out) {
        return;
    }

    // Iterate over input channels and output channels
    int c_out = blockIdx.w; // Output channel
    int c_in_start = (c_out * in_channels) / out_channels;
    int c_in_end = ((c_out + 1) * in_channels) / out_channels;

    float result = 0.0f;

    // Compute the 3D transposed convolution for each region
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            for (int k_h = 0; k_h < kernel_height; ++k_h) {
                for (int k_w = 0; k_w < kernel_width; ++k_w) {
                    int d_in = d_out * stride_depth - padding_depth + k_d;
                    int h_in = h_out * stride_height - padding_height + k_h;
                    int w_in = w_out * stride_width - padding_width + k_w;

                    if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                        int input_idx = ((b * in_channels + c_in) * depth_in + d_in) * height_in * width_in + h_in * width_in + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_depth + k_d) * kernel_height * kernel_width + k_h * kernel_width + k_w;
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if applicable
    if (bias) {
        result += bias[c_out];
    }

    int output_idx = ((b * out_channels + c_out) * depth_out + d_out) * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = result;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    std::tuple<int, int, int> kernel_size, 
    std::tuple<int, int, int> stride, 
    std::tuple<int, int, int> padding, 
    std::tuple<int, int, int> output_padding, 
    int groups) 
{
    // Input dimensions
    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int depth_in = input_size[2];
    int height_in = input_size[3];
    int width_in = input_size[4];

    // Output dimensions
    int kernel_depth, kernel_height, kernel_width;
    std::tie(kernel_depth, kernel_height, kernel_width) = kernel_size;
    int stride_depth, stride_height, stride_width;
    std::tie(stride_depth, stride_height, stride_width) = stride;
    int padding_depth, padding_height, padding_width;
    std::tie(padding_depth, padding_height, padding_width) = padding;
    int output_padding_depth, output_padding_height, output_padding_width;
    std::tie(output_padding_depth, output_padding_height, output_padding_width) = output_padding;

    int depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, in_channels, depth_out, height_out, width_out}, input.options());

    // Launch kernel
    dim3 threads_per_block(width_out);
    dim3 num_blocks(batch_size, depth_out, height_out, in_channels);
    
    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        in_channels,  // Out channels
        depth_in, 
        height_in, 
        width_in, 
        depth_out, 
        height_out, 
        width_out, 
        kernel_depth, 
        kernel_height, 
        kernel_width, 
        stride_depth, 
        stride_height, 
        stride_width, 
        padding_depth, 
        padding_height, 
        padding_width, 
        output_padding_depth, 
        output_padding_height, 
        output_padding_width, 
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_cuda", &conv_transpose3d_cuda, "3D transposed convolution");
}