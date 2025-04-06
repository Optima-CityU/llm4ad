#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    const int batch_size, 
    const int in_channels, 
    const int out_channels, 
    const int height_in, 
    const int width_in, 
    const int height_out, 
    const int width_out, 
    const int kernel_height, 
    const int kernel_width, 
    const int stride_height, 
    const int stride_width, 
    const int padding_height, 
    const int padding_width, 
    const int output_padding_height, 
    const int output_padding_width, 
    const int dilation_height, 
    const int dilation_width, 
    const int groups) 
{
    int batch_idx = blockIdx.x;
    int output_channel = blockIdx.y;
    int output_row = blockIdx.z / width_out;
    int output_col = blockIdx.z % width_out;

    if (batch_idx >= batch_size || output_channel >= out_channels || output_row >= height_out || output_col >= width_out) {
        return;
    }

    float value = 0.0f;

    int kernel_row_start = max(0, output_row * stride_height - padding_height);
    int kernel_col_start = max(0, output_col * stride_width - padding_width);

    for (int input_channel = 0; input_channel < in_channels; ++input_channel) {
        for (int k_row = 0; k_row < kernel_height; ++k_row) {
            for (int k_col = 0; k_col < kernel_width; ++k_col) {
                int input_row = kernel_row_start + k_row * dilation_height;
                int input_col = kernel_col_start + k_col * dilation_width;

                if (input_row < height_in && input_col < width_in) {
                    value += input[batch_idx * in_channels * height_in * width_in + input_channel * height_in * width_in + input_row * width_in + input_col] *
                             weight[output_channel * in_channels * kernel_height * kernel_width + input_channel * kernel_height * kernel_width + k_row * kernel_width + k_col];
                }
            }
        }
    }

    value += bias[output_channel];

    output[batch_idx * out_channels * height_out * width_out + output_channel * height_out * width_out + output_row * width_out + output_col] = value;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    std::tuple<int, int> kernel_size, 
    std::tuple<int, int> stride, 
    std::tuple<int, int> padding, 
    std::tuple<int, int> output_padding, 
    std::tuple<int, int> dilation, 
    int groups) 
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_height = std::get<0>(kernel_size);
    auto kernel_width = std::get<1>(kernel_size);

    auto stride_height = std::get<0>(stride);
    auto stride_width = std::get<1>(stride);

    auto padding_height = std::get<0>(padding);
    auto padding_width = std::get<1>(padding);

    auto output_padding_height = std::get<0>(output_padding);
    auto output_padding_width = std::get<1>(output_padding);

    auto dilation_height = std::get<0>(dilation);
    auto dilation_width = std::get<1>(dilation);

    auto height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    auto width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;

    torch::Tensor output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    int threads_per_block = 256;
    dim3 block(32, out_channels, threads_per_block);
    dim3 grid(batch_size, out_channels, height_out * width_out);

    conv_transpose2d_kernel<<<grid, block>>>( 
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        height_in, 
        width_in, 
        height_out, 
        width_out, 
        kernel_height, 
        kernel_width, 
        stride_height, 
        stride_width, 
        padding_height, 
        padding_width, 
        output_padding_height, 
        output_padding_width, 
        dilation_height, 
        dilation_width, 
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d", &conv_transpose2d_cuda, "2D Transposed Convolution (CUDA)");
}