#include <torch/extension.h>

__global__ void avg_pool1d_kernel(
    const float *input, float *output, 
    int batch_size, int in_channels, int input_length,
    int kernel_size, int stride, int padding, int output_length
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = threadIdx.x;

    if (batch_idx < batch_size && channel_idx < in_channels && output_idx < output_length) {
        int start_idx = output_idx * stride - padding;
        int end_idx = start_idx + kernel_size;
        start_idx = max(start_idx, 0);
        end_idx = min(end_idx, input_length);

        float sum = 0.0f;
        int count = 0;

        for (int i = start_idx; i < end_idx; i++) {
            sum += input[batch_idx * in_channels * input_length + channel_idx * input_length + i];
            count++;
        }

        output[batch_idx * in_channels * output_length + channel_idx * output_length + output_idx] = sum / count;
    }
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto output = torch::empty({input.size(0), input.size(1), 
                                (input.size(2) + 2 * padding - kernel_size) / stride + 1}, 
                                input.options());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);

    dim3 threads(output_length);
    dim3 blocks(batch_size, in_channels);

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        batch_size, in_channels, input_length,
        kernel_size, stride, padding, output_length
    );

    return output;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("avg_pool1d_cuda", &avg_pool1d_cuda);
}