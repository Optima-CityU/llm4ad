#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>

// CUDA kernel for depthwise 2D convolution.
// Assumes weight layout: [channels, 1, kernel_size, kernel_size] flattened to [channels * kernel_size * kernel_size].
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding)
{
    // Each block processes one (batch, channel) pair.
    int b = blockIdx.x;
    int c = blockIdx.y;
    int total_pixels = out_height * out_width;
    
    // Each thread computes multiple output pixels.
    for (int index = threadIdx.x; index < total_pixels; index += blockDim.x) {
        int out_row = index / out_width;
        int out_col = index % out_width;
        int in_row_start = out_row * stride - padding;
        int in_col_start = out_col * stride - padding;
        float sum = 0.0f;
        // Iterate over the kernel.
        for (int k_row = 0; k_row < kernel_size; k_row++) {
            for (int k_col = 0; k_col < kernel_size; k_col++) {
                int in_row = in_row_start + k_row;
                int in_col = in_col_start + k_col;
                if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                    int input_index = ((b * channels + c) * in_height + in_row) * in_width + in_col;
                    int weight_index = ((c * kernel_size + k_row) * kernel_size) + k_col;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        int output_index = ((b * channels + c) * out_height + out_row) * out_width + out_col;
        output[output_index] = sum + bias[c];
    }
}

torch::Tensor depthwise_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int stride,
    int padding)
{
    // If bias is not provided, create a zero bias tensor.
    torch::Tensor bias = bias_opt.has_value() ? bias_opt.value() :
        torch::zeros({input.size(1)}, input.options());

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int kernel_size = weight.size(2); // Assuming square kernel and weight shape [channels, 1, k, k]
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    // Launch configuration:
    // Each block is responsible for one (batch, channel) pair.
    // Threads within the block process different output pixels.
    dim3 blocks(batch_size, channels);
    int threads = 256;
    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d, "Depthwise Conv2d forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"));
}