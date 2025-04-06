#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D transposed convolution (conv_transpose1d)
template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    // Each thread computes one output element.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_length;
    if (index < total_elements) {
        // Calculate indices: l_out, output channel (oc) and batch index (b)
        int l_out = index % output_length;
        int oc = (index / output_length) % out_channels;
        int b = index / (out_channels * output_length);

        // Determine group information
        int out_channels_per_group = out_channels / groups;
        int group = oc / out_channels_per_group;
        int in_channels_per_group = in_channels / groups;

        scalar_t sum = 0;
        // Iterate over the input channels in the same group
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            int real_ic = group * in_channels_per_group + ic;
            // Iterate over the kernel elements
            for (int k = 0; k < kernel_size; k++) {
                // Calculate corresponding input index:
                // l_out = l_in * stride - padding + k  -->  l_in = (l_out + padding - k) / stride
                int l_in_temp = l_out + padding - k;
                if (l_in_temp % stride == 0) {
                    int l_in = l_in_temp / stride;
                    if (l_in >= 0 && l_in < input_length) {
                        // Weight layout: (in_channels, out_channels_per_group, kernel_size)
                        int weight_index = real_ic * (out_channels_per_group * kernel_size)
                                           + (oc - group * out_channels_per_group) * kernel_size + k;
                        int input_index = b * (in_channels * input_length)
                                          + real_ic * input_length + l_in;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
        // Add bias for the output channel
        sum += bias[oc];
        int output_index = b * (out_channels * output_length) + oc * output_length + l_out;
        output[output_index] = sum;
    }
}

at::Tensor conv_transpose1d_cuda_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    // Input shape: (batch_size, in_channels, input_length)
    auto batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    
    // Weight shape: (in_channels, out_channels_per_group, kernel_size)
    int out_channels_per_group = weight.size(1);
    int kernel_size = weight.size(2);
    int out_channels = out_channels_per_group * groups;
    
    // Compute output length as defined by transposed convolution
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = at::zeros({batch_size, out_channels, output_length}, input.options());

    int total_elements = batch_size * out_channels * output_length;
    const int threads = 1024;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_cuda_forward", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose1d_cuda_forward, "Conv Transpose1d forward (CUDA)");
}