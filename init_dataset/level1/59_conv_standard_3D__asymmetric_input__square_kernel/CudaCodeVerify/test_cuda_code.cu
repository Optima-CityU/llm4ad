#include <torch/extension.h>

__global__ void conv3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    const int batch_size, 
    const int in_channels, 
    const int out_channels, 
    const int height, 
    const int width, 
    const int depth, 
    const int height_out, 
    const int width_out, 
    const int depth_out, 
    const int stride, 
    const int padding, 
    const int dilation, 
    const int groups) 
{
    int n = blockIdx.x;  // batch index
    int oc = blockIdx.y;  // output channel index
    int oh = blockIdx.z;  // output height index
    int ow = threadIdx.x; // output width index
    int od = threadIdx.y; // output depth index

    int ih = oh * stride - padding;
    int iw = ow * stride - padding;
    int id = od * stride - padding;

    if (ih < 0 || ih >= height || iw < 0 || iw >= width || id < 0 || id >= depth) {
        return;
    }

    float result = 0.0f;
    int input_offset = n * in_channels * height * width * depth + ih * width * depth + iw * depth + id;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < weight_size_h; kh++) {
            for (int kw = 0; kw < weight_size_w; kw++) {
                for (int kd = 0; kd < weight_size_d; kd++) {
                    int h_offset = ih + kh * dilation;
                    int w_offset = iw + kw * dilation;
                    int d_offset = id + kd * dilation;
                    
                    if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width && d_offset >= 0 && d_offset < depth) {
                        int input_idx = n * in_channels * height * width * depth + ic * height * width * depth + h_offset * width * depth + w_offset * depth + d_offset;
                        int weight_idx = oc * in_channels * weight_size_h * weight_size_w * weight_size_d + ic * weight_size_h * weight_size_w * weight_size_d + kh * weight_size_w * weight_size_d + kw * weight_size_d + kd;
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        result += bias[oc];
    }

    output[n * out_channels * height_out * width_out * depth_out + oc * height_out * width_out * depth_out + oh * width_out * depth_out + ow * depth_out + od] = result;
}

torch::Tensor conv3d_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding, 
    int dilation, 
    int groups) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);
    const int depth = input.size(4);
    
    const int height_out = (height + 2 * padding - dilation * (weight.size(2) - 1) - 1) / stride + 1;
    const int width_out = (width + 2 * padding - dilation * (weight.size(3) - 1) - 1) / stride + 1;
    const int depth_out = (depth + 2 * padding - dilation * (weight.size(4) - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out, depth_out}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(batch_size, out_channels, height_out);
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, in_channels, out_channels, 
        height, width, depth, 
        height_out, width_out, depth_out, 
        stride, padding, dilation, groups
    );

    return output;
}