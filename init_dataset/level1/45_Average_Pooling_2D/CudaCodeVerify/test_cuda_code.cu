#include <torch/extension.h>

__global__ void avg_pool2d_kernel(
    const float* input, 
    float* output, 
    const int batch_size, 
    const int channels, 
    const int height, 
    const int width, 
    const int kernel_size, 
    const int stride, 
    const int padding, 
    const int output_height, 
    const int output_width) 
{
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int out_h = blockIdx.z / output_width;
    int out_w = blockIdx.z % output_width;

    int h_start = out_h * stride - padding;
    int w_start = out_w * stride - padding;
    int h_end = min(h_start + kernel_size, height + padding);
    int w_end = min(w_start + kernel_size, width + padding);

    int count = 0;
    float sum = 0.0f;

    for (int h = max(h_start, 0); h < min(h_end, height); ++h) {
        for (int w = max(w_start, 0); w < min(w_end, width); ++w) {
            sum += input[batch_idx * channels * height * width + channel_idx * height * width + h * width + w];
            count++;
        }
    }

    output[batch_idx * channels * output_height * output_width + channel_idx * output_height * output_width + out_h * output_width + out_w] = sum / count;
}

torch::Tensor avg_pool2d_forward(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto output_height = (input.size(2) + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input.size(3) + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({input.size(0), input.size(1), output_height, output_width}, input.options());

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    dim3 block_dim(32, 32);
    dim3 grid_dim(batch_size, channels, output_height * output_width);

    avg_pool2d_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size, stride, padding, output_height, output_width);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling Forward Pass");
}