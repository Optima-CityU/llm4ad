#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void avg_pool3d_kernel(
    const float* input, float* output,
    int batch_size, int channels, int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int output_depth, int output_height, int output_width) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int depth_idx = threadIdx.x;
    int height_idx = threadIdx.y;
    int width_idx = threadIdx.z;

    if (depth_idx < output_depth && height_idx < output_height && width_idx < output_width) {
        int start_d = depth_idx * stride - padding;
        int start_h = height_idx * stride - padding;
        int start_w = width_idx * stride - padding;

        float sum = 0.0f;
        int count = 0;

        for (int dz = 0; dz < kernel_size; dz++) {
            for (int dh = 0; dh < kernel_size; dh++) {
                for (int dw = 0; dw < kernel_size; dw++) {
                    int d = start_d + dz;
                    int h = start_h + dh;
                    int w = start_w + dw;

                    if (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 && w < width) {
                        int input_idx = (batch_idx * channels + channel_idx) * depth * height * width + d * height * width + h * width + w;
                        sum += input[input_idx];
                        count++;
                    }
                }
            }
        }

        if (count > 0) {
            int output_idx = (batch_idx * channels + channel_idx) * output_depth * output_height * output_width + depth_idx * output_height * output_width + height_idx * output_width + width_idx;
            output[output_idx] = sum / count;
        }
    }
}

void avg_pool3d_forward_cuda(
    at::Tensor input, at::Tensor output,
    int kernel_size, int stride, int padding) {

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int output_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;

    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(batch_size, channels);

    avg_pool3d_kernel<<<grid_dim, block_dim>>>(
        input.data<float>(), output.data<float>(),
        batch_size, channels, depth, height, width,
        kernel_size, stride, padding,
        output_depth, output_height, output_width
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool3d_forward_cuda, "3D Average Pooling Forward (CUDA)");
}