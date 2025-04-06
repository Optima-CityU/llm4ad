// conv2d_kernel.cu

#include <torch/extension.h>

__global__ void conv2d_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, 
    int batch_size, int in_channels, int out_channels, 
    int height, int width, int kernel_h, int kernel_w, 
    int stride, int padding, int dilation, int groups, 
    int out_height, int out_width) 
{
    int batch_index = blockIdx.x;
    int channel_index = blockIdx.y;
    int row_index = threadIdx.y + blockIdx.z * blockDim.y;
    int col_index = threadIdx.x + blockIdx.w * blockDim.x;

    if (row_index >= out_height || col_index >= out_width) return;

    int h_start = row_index * stride - padding;
    int w_start = col_index * stride - padding;

    float sum = 0.0f;

    for (int i = 0; i < in_channels / groups; ++i) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = h_start + kh * dilation;
                int w = w_start + kw * dilation;

                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_index = ((batch_index * in_channels + (i + channel_index * in_channels / groups)) * height + h) * width + w;
                    int weight_index = ((channel_index * in_channels / groups + i) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
    }

    sum += bias[channel_index];
    output[(batch_index * out_channels + channel_index) * out_height * out_width + row_index * out_width + col_index] = sum;
}

torch::Tensor conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    int stride, int padding, int dilation, int groups) 
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_height = (height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(batch_size, out_channels, (out_height + 15) / 16, (out_width + 15) / 16);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, 
        height, width, kernel_h, kernel_w, stride, padding, dilation, groups, 
        out_height, out_width
    );

    return output;
}

TORCH_LIBRARY(myconv2d, m) {
  m.def("forward", &conv2d_forward);
}