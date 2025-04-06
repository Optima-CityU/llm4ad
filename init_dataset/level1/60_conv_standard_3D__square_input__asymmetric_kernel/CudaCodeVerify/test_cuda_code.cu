// Import required libraries
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth_in, 
    int height_in, 
    int width_in, 
    int depth_out, 
    int height_out, 
    int width_out, 
    int stride, 
    int padding, 
    int dilation, 
    int groups) 
{
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int d_out = blockIdx.z / (width_out * height_out);
    int h_out = (blockIdx.z % (width_out * height_out)) / width_out;
    int w_out = blockIdx.z % width_out;
    
    int d_in = d_out * stride - padding;
    int h_in = h_out * stride - padding;
    int w_in = w_out * stride - padding;

    if (d_in < 0 || h_in < 0 || w_in < 0 || d_in >= depth_in || h_in >= height_in || w_in >= width_in) {
        return; // Skip if the position is out of bounds
    }

    float value = 0.0;
    
    // Convolution calculation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < weight[0]; ++k_d) {  // Assuming weight is 4D [out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]
            for (int k_h = 0; k_h < weight[1]; ++k_h) {
                for (int k_w = 0; k_w < weight[2]; ++k_w) {
                    int d_idx = d_in + k_d * dilation;
                    int h_idx = h_in + k_h * dilation;
                    int w_idx = w_in + k_w * dilation;

                    if (d_idx >= 0 && h_idx >= 0 && w_idx >= 0 && d_idx < depth_in && h_idx < height_in && w_idx < width_in) {
                        value += input[batch_idx * in_channels * depth_in * height_in * width_in + c_in * depth_in * height_in * width_in + d_idx * height_in * width_in + h_idx * width_in + w_idx] 
                               * weight[out_channel * in_channels * weight[3] * weight[4] + c_in * weight[3] * weight[4] + k_d * weight[4] + k_h * weight[4] + k_w];
                    }
                }
            }
        }
    }

    // Add the bias if applicable
    value += bias[out_channel];

    // Store the output
    output[batch_idx * out_channels * depth_out * height_out * width_out + out_channel * depth_out * height_out * width_out + d_out * height_out * width_out + h_out * width_out + w_out] = value;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding, 
    int dilation, 
    int groups) 
{
    // Getting dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    // Calculate output dimensions
    int depth_out = (depth_in + 2 * padding - kernel_depth) / stride + 1;
    int height_out = (height_in + 2 * padding - kernel_height) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_width) / stride + 1;

    // Allocate memory for the output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    // Kernel launch parameters
    dim3 blocks(batch_size, out_channels, depth_out * height_out * width_out);
    dim3 threads(1);

    // Launch the kernel
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        depth_in, 
        height_in, 
        width_in, 
        depth_out, 
        height_out, 
        width_out, 
        stride, 
        padding, 
        dilation, 
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_cuda", &conv3d_cuda, "3D Convolution (CUDA)");
}