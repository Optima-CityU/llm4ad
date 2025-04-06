
```C++
#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float *input, const float *weight, const float *bias,
    float *output,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in, int height_out, int width_out,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, int dilation_h, int dilation_w,
    int groups) {

    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int row_idx = blockIdx.z % height_out;
    int col_idx = blockIdx.z / height_out;

    int in_channel_idx = threadIdx.x;

    if (batch_idx < batch_size && out_channel_idx < out_channels) {
        for (int i = 0; i < in_channels; ++i) {
            int input_row_start = row_idx * stride_h - padding_h;
            int input_col_start = col_idx * stride_w - padding_w;

            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int in_row = input_row_start + ky * dilation_h;
                    int in_col = input_col_start + kx * dilation_w;

                    if (in_row >= 0 && in_row < height_in && in_col >= 0 && in_col < width_in) {
                        int input_idx = ((batch_idx * in_channels + i) * height_in + in_row) * width_in + in_col;
                        int weight_idx = (((out_channel_idx * in_channels + i) / groups) * kernel_h + ky) * kernel_w + kx;
                        int output_idx = ((batch_idx * out_channels + out_channel_idx) * height_out + row_idx) * width_out + col_idx;

                        output[output_idx] += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            int output_idx = ((batch_idx * out_channels + out_channel_idx) * height_out + row_idx) * width_out + col_idx;
            output[output_idx] += bias[out_channel_idx];
        }
    }
}

void conv_transpose2d_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int height_out = output.size(2);
    const int width_out = output.size(3);

    dim3 block_dim(in_channels, 1, 1);
    dim3 grid_dim(batch_size, out_channels, height_out * width_out);

    conv_transpose2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height_in, width_in, height_out, width_out,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w,
        groups
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_cuda", &conv_transpose2d_cuda, "2D Transposed Convolution (CUDA)");
}
