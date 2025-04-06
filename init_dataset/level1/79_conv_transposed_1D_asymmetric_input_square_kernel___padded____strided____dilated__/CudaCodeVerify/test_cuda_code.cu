
```C++
// Include pybind11 and necessary headers
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv1d_kernel(
    const float *x, const float *weight, const float *bias, 
    float *output, int batch_size, int in_channels, int out_channels, 
    int length, int kernel_size, int stride, int padding, int dilation, int length_out) {

    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int i = threadIdx.x + blockIdx.z * blockDim.x;

    if (i < length_out) {
        float value = 0.0f;
        
        for (int j = 0; j < in_channels; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_idx = i * stride - padding + k * dilation;
                if (input_idx >= 0 && input_idx < length) {
                    value += weight[out_channel_idx * in_channels * kernel_size + j * kernel_size + k] * 
                             x[batch_idx * in_channels * length + j * length + input_idx];
                }
            }
        }

        value += bias[out_channel_idx];
        output[batch_idx * out_channels * length_out + out_channel_idx * length_out + i] = value;
    }
}

void transposed_conv1d_cuda(
    at::Tensor x, at::Tensor weight, at::Tensor bias, 
    at::Tensor output, int stride, int padding, int dilation) {

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int out_channels = weight.size(0);
    int length = x.size(2);
    int kernel_size = weight.size(2);
    int length_out = (length - 1) * stride + kernel_size - 2 * padding;

    dim3 threads_per_block(256);
    dim3 blocks_per_grid(batch_size, out_channels, (length_out + 255) / 256);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "transposed_conv1d_cuda", ([&] {
        transposed_conv1d_kernel<<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
            output.data_ptr<float>(), batch_size, in_channels, out_channels, 
            length, kernel_size, stride, padding, dilation, length_out
        );
    }));

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transposed_conv1d_cuda", &transposed_conv1d_cuda, "Transposed 1D Convolution (CUDA)");
}
