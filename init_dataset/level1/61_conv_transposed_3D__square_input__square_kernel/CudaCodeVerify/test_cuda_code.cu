// Import necessary PyTorch/CUDA libraries
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_channels,
    const int64_t out_channels,
    const int64_t depth_in,
    const int64_t height_in,
    const int64_t width_in,
    const int64_t depth_out,
    const int64_t height_out,
    const int64_t width_out,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride,
    const int64_t padding,
    const int64_t output_padding,
    const int64_t groups) {

    // Calculate the indices
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * depth_out * height_out * width_out) {
        return;
    }

    // Determine batch, output channel, and output coordinates
    int batch_index = index / (out_channels * depth_out * height_out * width_out);
    int out_channel_index = (index / (depth_out * height_out * width_out)) % out_channels;
    int depth_out_index = (index / (height_out * width_out)) % depth_out;
    int height_out_index = (index / width_out) % height_out;
    int width_out_index = index % width_out;

    // Calculate the corresponding input coordinates
    int depth_in_index = depth_out_index * stride - padding;
    int height_in_index = height_out_index * stride - padding;
    int width_in_index = width_out_index * stride - padding;

    // Apply output padding
    depth_in_index += (depth_out_index < output_padding) ? 1 : 0;
    height_in_index += (height_out_index < output_padding) ? 1 : 0;
    width_in_index += (width_out_index < output_padding) ? 1 : 0;

    // Calculate the start and end bounds for the kernel in each dimension
    int depth_start = max(0, depth_in_index);
    int depth_end = min(depth_in_index + kernel_d, depth_in);
    int height_start = max(0, height_in_index);
    int height_end = min(height_in_index + kernel_h, height_in);
    int width_start = max(0, width_in_index);
    int width_end = min(width_in_index + kernel_w, width_in);

    // Compute the convolution output
    float value = 0.0f;
    for (int c = 0; c < in_channels / groups; ++c) {
        for (int d = depth_start; d < depth_end; ++d) {
            for (int h = height_start; h < height_end; ++h) {
                for (int w = width_start; w < width_end; ++w) {
                    int input_index = (batch_index * in_channels * depth_in * height_in * width_in) + 
                                      (c + out_channel_index * in_channels / groups) * depth_in * height_in * width_in +
                                      d * height_in * width_in + h * width_in + w;

                    int weight_index = (out_channel_index * in_channels * kernel_d * kernel_h * kernel_w) + 
                                       c * kernel_d * kernel_h * kernel_w + 
                                       (d - depth_in_index) * kernel_h * kernel_w + 
                                       (h - height_in_index) * kernel_w + 
                                       (w - width_in_index);

                    value += input[input_index] * weight[weight_index];
                }
            }
        }
    }

    // Add bias and write to output
    if (bias != nullptr) {
        value += bias[out_channel_index];
    }

    output[index] = value;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    // Get input and output dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);

    // Calculate output dimensions
    int depth_out = (depth_in - 1) * stride - 2 * padding + kernel_d + output_padding;
    int height_out = (height_in - 1) * stride - 2 * padding + kernel_h + output_padding;
    int width_out = (width_in - 1) * stride - 2 * padding + kernel_w + output_padding;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    // Define block and grid sizes
    int block_size = 256;
    int grid_size = (batch_size * out_channels * depth_out * height_out * width_out + block_size - 1) / block_size;

    // Launch the CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_kernel", ([&] {
        conv_transpose3d_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
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
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
            groups);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_cuda", &conv_transpose3d_cuda, "3D Transposed Convolution (CUDA)");
}