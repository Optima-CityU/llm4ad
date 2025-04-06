#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    // Input dimensions
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    // Weight dimensions
    // weight shape: (in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w)
    const int out_channels,  // total output channels (i.e. groups * out_channels_per_group)
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    // Convolution parameters
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_pad_d,
    const int out_pad_h,
    const int out_pad_w,
    const int groups
) {
    // Compute output dimensions
    const int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    const int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    const int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

    // Total number of output elements
    const int total = batch_size * out_channels * out_depth * out_height * out_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    // Map flat index to (n, oc, od, oh, ow)
    int w_out = index % out_width;
    int tmp = index / out_width;
    int h_out = tmp % out_height;
    tmp = tmp / out_height;
    int d_out = tmp % out_depth;
    tmp = tmp / out_depth;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    scalar_t value = 0;

    // Determine group index and related channel ranges
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group_idx = oc / out_channels_per_group;
    int weight_oc = oc - group_idx * out_channels_per_group; // index within the group's output channels

    // Loop over the corresponding input channels for this group
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        // Global input channel index
        int input_channel = group_idx * in_channels_per_group + ic;
        // Loop over kernel dimensions
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Calculate the corresponding input index before checking stride division
                    int d_in_tmp = d_out + pad_d - kd;
                    int h_in_tmp = h_out + pad_h - kh;
                    int w_in_tmp = w_out + pad_w - kw;
                    
                    // Check if the computed indices are aligned with the stride
                    if (d_in_tmp % stride_d == 0 && h_in_tmp % stride_h == 0 && w_in_tmp % stride_w == 0) {
                        int d_in = d_in_tmp / stride_d;
                        int h_in = h_in_tmp / stride_h;
                        int w_in = w_in_tmp / stride_w;
                        // Validate that the indices are within bounds of the input dimensions
                        if (d_in >= 0 && d_in < in_depth &&
                            h_in >= 0 && h_in < in_height &&
                            w_in >= 0 && w_in < in_width) {
                            // Compute flat index for the input tensor: shape (batch_size, in_channels, in_depth, in_height, in_width)
                            int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                            // Compute flat index for the weight tensor:
                            // Weight shape: (in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w)
                            int weight_idx = ((((input_channel - group_idx * in_channels_per_group) * out_channels_per_group + weight_oc) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    // Add bias if provided
    if (bias != nullptr) {
        value += bias[oc];
    }
    // Write the computed value to the output tensor
    int output_idx = (((n * out_channels + oc) * out_depth + d_out) * out_height + h_out) * out_width + w_out;
    output[output_idx] = value;
}

torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int groups
) {
    // Input dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    // Weight dimensions: weight shape = (in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w)
    const int out_channels = weight.size(1) * groups;
    
    // Kernel size
    const int kernel_d = kernel_size[0];
    const int kernel_h = kernel_size[1];
    const int kernel_w = kernel_size[2];
    
    // Strides
    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];
    
    // Paddings
    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];
    
    // Output padding
    const int out_pad_d = output_padding[0];
    const int out_pad_h = output_padding[1];
    const int out_pad_w = output_padding[2];
    
    // Calculate output dimensions
    const int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    const int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    const int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    // Launch CUDA kernel
    const int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    const int threads = 1024;
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_forward_cuda", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_depth,
            in_height,
            in_width,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            out_pad_d,
            out_pad_h,
            out_pad_w,
            groups
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose3d_forward, "Transposed 3D convolution forward (CUDA)");
}