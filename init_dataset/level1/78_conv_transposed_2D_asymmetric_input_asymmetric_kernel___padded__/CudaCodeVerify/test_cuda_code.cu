// Include necessary headers
#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, int batch_size, int in_channels, int out_channels, 
    int height_in, int width_in, int height_out, int width_out, 
    int stride_h, int stride_w, int pad_h, int pad_w) 
{
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int out_c = threadIdx.x;

    if (batch_idx < batch_size && out_y < height_out && out_x < width_out && out_c < out_channels) {
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int kernel_y = 0; kernel_y < weight_height; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < weight_width; ++kernel_x) {
                    int in_y = out_y * stride_h - pad_h + kernel_y;
                    int in_x = out_x * stride_w - pad_w + kernel_x;

                    if (in_y >= 0 && in_y < height_in && in_x >= 0 && in_x < width_in) {
                        int input_idx = ((batch_idx * in_channels + in_c) * height_in + in_y) * width_in + in_x;
                        int weight_idx = ((out_c * in_channels + in_c) * weight_height + kernel_y) * weight_width + kernel_x;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        sum += bias[out_c];
        int output_idx = ((batch_idx * out_channels + out_c) * height_out + out_y) * width_out + out_x;
        output[output_idx] = sum;
    }
}

void conv_transpose2d_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias, 
    at::Tensor output, 
    int stride_h, int stride_w, int pad_h, int pad_w) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int height_out = output.size(2);
    const int width_out = output.size(3);

    dim3 block_size(out_channels);
    dim3 grid_size(batch_size, height_out, width_out);

    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), batch_size, in_channels, out_channels, 
        height_in, width_in, height_out, width_out, 
        stride_h, stride_w, pad_h, pad_w
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d", &conv_transpose2d_cuda, "2D Transposed Convolution (CUDA)");
}