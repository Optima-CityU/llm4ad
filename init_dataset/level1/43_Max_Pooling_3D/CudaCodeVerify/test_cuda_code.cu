// CUDA kernel for 3D max pooling operation
#include <torch/extension.h>

__global__ void max_pool3d_kernel(
    const float* input, 
    float* output, 
    const int* indices, 
    const int batch_size, 
    const int channels, 
    const int input_depth, 
    const int input_height, 
    const int input_width, 
    const int kernel_size, 
    const int stride, 
    const int padding, 
    const int dilation, 
    const bool return_indices, 
    const bool ceil_mode
) {
    // Calculate the output indices
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_d = threadIdx.x;
    int output_h = threadIdx.y;
    int output_w = threadIdx.z;

    // Calculate input dimensions after padding
    int input_d = output_d * stride - padding;
    int input_h = output_h * stride - padding;
    int input_w = output_w * stride - padding;

    // Check for padding and dilation
    if (input_d < 0 || input_d >= input_depth || input_h < 0 || input_h >= input_height || input_w < 0 || input_w >= input_width) {
        return;
    }

    // Loop over the kernel size
    float max_val = -INFINITY;
    int max_idx = -1;
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_idx = (batch_idx * channels + channel_idx) * (input_depth * input_height * input_width) 
                                 + (input_d + kd * dilation) * (input_height * input_width)
                                 + (input_h + kh * dilation) * input_width
                                 + (input_w + kw * dilation);

                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }

    // Store the result
    int output_idx = (batch_idx * channels + channel_idx) * (input_depth * input_height * input_width) 
                      + output_d * (input_height * input_width) 
                      + output_h * input_width 
                      + output_w;

    output[output_idx] = max_val;

    // Optionally, store the index if required
    if (return_indices) {
        indices[output_idx] = max_idx;
    }
}

void max_pool3d_forward(
    torch::Tensor input, 
    torch::Tensor output, 
    torch::Tensor indices, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation, 
    bool return_indices, 
    bool ceil_mode
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    // Launch CUDA kernel
    dim3 threads(16, 16, 16);
    dim3 blocks(batch_size, channels, input_depth);
    
    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        indices.data_ptr<int>(), 
        batch_size, channels, input_depth, input_height, input_width,
        kernel_size, stride, padding, dilation, return_indices, ceil_mode
    );
}