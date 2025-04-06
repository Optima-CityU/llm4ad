
```C++
// Includes for CUDA and PyTorch bindings
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth, 
    int height, 
    int width, 
    int stride, 
    int padding, 
    int output_padding, 
    int dilation, 
    int groups, 
    int depth_out, 
    int height_out, 
    int width_out
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x + blockIdx.z * blockDim.x;
    int h = threadIdx.y + blockIdx.z * blockDim.y;
    int w = threadIdx.z + blockIdx.z * blockDim.z;

    if (d < depth_out && h < height_out && w < width_out) {
        float sum = 0.0f;
        int start_d = d * stride - padding;
        int start_h = h * stride - padding;
        int start_w = w * stride - padding;

        for (int i = 0; i < in_channels; i++) {
            for (int kd = 0; kd < weight.shape[2]; kd++) {
                for (int kh = 0; kh < weight.shape[3]; kh++) {
                    for (int kw = 0; kw < weight.shape[4]; kw++) {
                        int in_d = start_d + kd * dilation;
                        int in_h = start_h + kh * dilation;
                        int in_w = start_w + kw * dilation;

                        if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                            int weight_idx = (c * in_channels + i) * weight.shape[2] * weight.shape[3] * weight.shape[4] + kd * weight.shape[3] * weight.shape[4] + kh * weight.shape[4] + kw;
                            int input_idx = b * in_channels * depth * height * width + i * depth * height * width + in_d * height * width + in_h * width + in_w;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias if applicable
        if (bias != nullptr) {
            sum += bias[c];
        }

        int output_idx = b * out_channels * depth_out * height_out * width_out + c * depth_out * height_out * width_out + d * height_out * width_out + h * width_out + w;
        output[output_idx] = sum;
    }
}

void conv_transpose3d_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor output, 
    int stride, 
    int padding, 
    int output_padding, 
    int dilation, 
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int depth_out = output.size(2);
    int height_out = output.size(3);
    int width_out = output.size(4);

    dim3 threads(8, 8, 8);
    dim3 blocks(batch_size, out_channels, (depth_out * height_out * width_out + 511) / 512);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        depth_out,
        height_out,
        width_out
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_forward", &conv_transpose3d_forward, "ConvTranspose3D forward (CUDA)");
}
