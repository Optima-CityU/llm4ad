#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    const int batch_size, 
    const int in_channels, 
    const int out_channels, 
    const int depth_in, 
    const int width_in, 
    const int height_in, 
    const int depth_out, 
    const int width_out, 
    const int height_out, 
    const int kernel_d, 
    const int kernel_w, 
    const int kernel_h, 
    const int stride, 
    const int padding, 
    const int dilation, 
    const int groups) 
{
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int d = threadIdx.x;
    int w = threadIdx.y;
    int h = threadIdx.z;

    if (b >= batch_size || oc >= out_channels || d >= depth_out || w >= width_out || h >= height_out) {
        return;
    }

    float value = 0.0f;
    int d_start = d * stride - padding;
    int w_start = w * stride - padding;
    int h_start = h * stride - padding;

    for (int ic = 0; ic < in_channels / groups; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    int d_in = d_start + kd * dilation;
                    int w_in = w_start + kw * dilation;
                    int h_in = h_start + kh * dilation;

                    if (d_in >= 0 && d_in < depth_in && w_in >= 0 && w_in < width_in && h_in >= 0 && h_in < height_in) {
                        int input_idx = ((b * in_channels + ic) * depth_in + d_in) * width_in * height_in + w_in * height_in + h_in;
                        int weight_idx = ((oc * in_channels / groups + ic) * kernel_d + kd) * kernel_w * kernel_h + kw * kernel_h + kh;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    value += bias[oc];
    int output_idx = ((b * out_channels + oc) * depth_out + d) * width_out * height_out + w * height_out + h;
    output[output_idx] = value;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding, 
    int dilation, 
    int groups) 
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int depth_in = input.size(2);
    int width_in = input.size(3);
    int height_in = input.size(4);
    int depth_out = (depth_in + 2 * padding - weight.size(2)) / stride + 1;
    int width_out = (width_in + 2 * padding - weight.size(3)) / stride + 1;
    int height_out = (height_in + 2 * padding - weight.size(4)) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, out_channels, depth_out, width_out, height_out}, input.options());

    dim3 threads(8, 8, 8);
    dim3 blocks(batch_size, out_channels, 1);

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        depth_in, 
        width_in, 
        height_in, 
        depth_out, 
        width_out, 
        height_out, 
        weight.size(2), 
        weight.size(3), 
        weight.size(4), 
        stride, 
        padding, 
        dilation, 
        groups);

    return output;
}