// cuda_conv3d_kernel.cu

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int depth_in, int height_in, int width_in,
    int depth_out, int height_out, int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int b = blockIdx.x;  // batch index
    int c_out = blockIdx.y;  // output channel index
    int d_out = blockIdx.z % depth_out;
    int h_out = (blockIdx.z / depth_out) % height_out;
    int w_out = blockIdx.z / (depth_out * height_out);

    int d_start = d_out * stride_d - padding_d;
    int h_start = h_out * stride_h - padding_h;
    int w_start = w_out * stride_w - padding_w;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int k_d = 0; k_d < kernel_d; k_d++) {
            for (int k_h = 0; k_h < kernel_h; k_h++) {
                for (int k_w = 0; k_w < kernel_w; k_w++) {
                    int d_idx = d_start + k_d * dilation_d;
                    int h_idx = h_start + k_h * dilation_h;
                    int w_idx = w_start + k_w * dilation_w;

                    if (d_idx >= 0 && d_idx < depth_in && h_idx >= 0 && h_idx < height_in && w_idx >= 0 && w_idx < width_in) {
                        int input_idx = ((b * in_channels + c_in) * depth_in + d_idx) * height_in * width_in + h_idx * width_in + w_idx;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_d + k_d) * kernel_h * kernel_w + k_h * kernel_w + k_w;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    if (bias) {
        sum += bias[c_out];
    }

    int output_idx = ((b * out_channels + c_out) * depth_out + d_out) * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = sum;
}

at::Tensor conv3d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    std::tuple<int, int, int> stride,
    std::tuple<int, int, int> padding,
    std::tuple<int, int, int> dilation,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth_in = input.size(2);
    const int height_in = input.size(3);
    const int width_in = input.size(4);

    const int out_channels = weight.size(0);
    const int depth_out = (depth_in + 2 * std::get<0>(padding) - std::get<0>(dilation) * (weight.size(2) - 1) - 1) / std::get<0>(stride) + 1;
    const int height_out = (height_in + 2 * std::get<1>(padding) - std::get<1>(dilation) * (weight.size(3) - 1) - 1) / std::get<1>(stride) + 1;
    const int width_out = (width_in + 2 * std::get<2>(padding) - std::get<2>(dilation) * (weight.size(4) - 1) - 1) / std::get<2>(stride) + 1;

    at::Tensor output = at::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    dim3 block_dim(1, 1, 1);
    dim3 grid_dim(batch_size, out_channels, depth_out * height_out * width_out);

    conv3d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        weight.size(2), weight.size(3), weight.size(4),
        std::get<0>(stride), std::get<1>(stride), std::get<2>(stride),
        std::get<0>(padding), std::get<1>(padding), std::get<2>(padding),
        std::get<0>(dilation), std::get<1>(dilation), std::get<2>(dilation),
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_cuda, "3D Convolution (CUDA)");
}