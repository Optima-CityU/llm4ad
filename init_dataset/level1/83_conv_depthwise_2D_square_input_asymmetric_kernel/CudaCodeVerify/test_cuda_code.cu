#include <torch/extension.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_channels, 
    int height, 
    int width, 
    int out_height, 
    int out_width, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation
) {
    int b = blockIdx.x;  // Batch index
    int c = blockIdx.y;  // Channel index
    int h = blockIdx.z / out_width;  // Output height index
    int w = blockIdx.z % out_width;  // Output width index

    int input_h = h * stride - padding; 
    int input_w = w * stride - padding;

    if (input_h < 0 || input_w < 0 || input_h >= height || input_w >= width) {
        return;
    }

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = input_h + kh * dilation;
            int iw = input_w + kw * dilation;

            if (ih >= 0 && iw >= 0 && ih < height && iw < width) {
                int input_index = b * in_channels * height * width + c * height * width + ih * width + iw;
                int weight_index = c * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[input_index] * weight[weight_index];
            }
        }
    }

    int output_index = b * in_channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
    output[output_index] = sum + bias[c];
}

void depthwise_conv2d_forward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor output, 
    int stride, 
    int padding, 
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_size = weight.size(2);  // Assuming square kernel (height == width)
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    dim3 block_size(1, 1, 256);  // Use 256 threads per block for 2D convolution
    dim3 grid_size(batch_size, in_channels, out_height * out_width);

    depthwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        height, 
        width, 
        out_height, 
        out_width, 
        kernel_size, 
        stride, 
        padding, 
        dilation
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_forward, "Depthwise 2D Convolution (CUDA)");
}