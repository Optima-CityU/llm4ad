#include <torch/extension.h>

__global__ void conv1d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels, 
    int length, int kernel_size, int stride, int padding, int dilation, int groups) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = threadIdx.x;

    int output_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (batch_idx < batch_size && channel_idx < out_channels && output_idx < output_length) {
        int input_start = output_idx * stride - padding;
        int input_end = input_start + kernel_size * dilation;

        float result = 0.0f;
        for (int i = input_start; i < input_end; i += dilation) {
            if (i >= 0 && i < length) {
                result += input[batch_idx * in_channels * length + channel_idx * length + i] *
                          weight[channel_idx * in_channels * kernel_size + (i - input_start) / dilation];
            }
        }

        result += bias[channel_idx];
        output[batch_idx * out_channels * output_length + channel_idx * output_length + output_idx] = result;
    }
}

torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                             int stride, int padding, int dilation, int groups) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    torch::Tensor output = torch::zeros({batch_size, out_channels, 
                                        (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1},
                                        input.options());

    dim3 threads(256);
    dim3 blocks(batch_size, out_channels, 1);
    conv1d_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), 
                                       bias.data_ptr<float>(), output.data_ptr<float>(),
                                       batch_size, in_channels, out_channels, length, 
                                       kernel_size, stride, padding, dilation, groups);
    
    return output;
}