// module_fn CUDA kernel for transposed 3D convolution
#include <torch/extension.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    const int batch_size, 
    const int in_channels, 
    const int depth_in, 
    const int height_in, 
    const int width_in, 
    const int out_channels, 
    const int depth_out, 
    const int height_out, 
    const int width_out, 
    const int kernel_d, 
    const int kernel_h, 
    const int kernel_w, 
    const int stride_d, 
    const int stride_h, 
    const int stride_w, 
    const int padding_d, 
    const int padding_h, 
    const int padding_w, 
    const int output_padding_d, 
    const int output_padding_h, 
    const int output_padding_w, 
    const int groups) 
{
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int d_out = threadIdx.z;
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;

    if (d_out >= depth_out || h_out >= height_out || w_out >= width_out) {
        return;
    }

    int depth_in_start = d_out * stride_d - padding_d;
    int height_in_start = h_out * stride_h - padding_h;
    int width_in_start = w_out * stride_w - padding_w;

    float val = 0.0f;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int d_k = 0; d_k < kernel_d; d_k++) {
            for (int h_k = 0; h_k < kernel_h; h_k++) {
                for (int w_k = 0; w_k < kernel_w; w_k++) {
                    int d_in = depth_in_start + d_k;
                    int h_in = height_in_start + h_k;
                    int w_in = width_in_start + w_k;

                    if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                        int weight_idx = (((out_channel_idx * in_channels + c_in) * kernel_d + d_k) * kernel_h + h_k) * kernel_w + w_k;
                        int input_idx = ((batch_idx * in_channels + c_in) * depth_in + d_in) * height_in * width_in + h_in * width_in + w_in;

                        val += weight[weight_idx] * input[input_idx];
                    }
                }
            }
        }
    }

    val += bias[out_channel_idx];

    int output_idx = ((batch_idx * out_channels + out_channel_idx) * depth_out + d_out) * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = val;
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
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    auto kernel_d = std::get<0>(kernel_size);
    auto kernel_h = std::get<1>(kernel_size);
    auto kernel_w = std::get<2>(kernel_size);

    auto stride_d = std::get<0>(stride);
    auto stride_h = std::get<1>(stride);
    auto stride_w = std::get<2>(stride);

    auto padding_d = std::get<0>(padding);
    auto padding_h = std::get<1>(padding);
    auto padding_w = std::get<2>(padding);

    auto output_padding_d = std::get<0>(output_padding);
    auto output_padding_h = std::get<1>(output_padding);
    auto output_padding_w = std::get<2>(output_padding);

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, weight.size(0), depth_out, height_out, width_out}, input.options());

    dim3 threads_per_block(8, 8, 8);
    dim3 num_blocks(batch_size, weight.size(0), 1);

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        depth_in, 
        height_in, 
        width_in, 
        weight.size(0), 
        depth_out, 
        height_out, 
        width_out, 
        kernel_d, 
        kernel_h, 
        kernel_w, 
        stride_d, 
        stride_h, 
        stride_w, 
        padding_d, 
        padding_h, 
        padding_w, 
        output_padding_d, 
        output_padding_h, 
        output_padding_w, 
        groups
    );

    return output;
}