
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUDA kernel for 2D max pooling 
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int pooled_height,
    const int pooled_width)
{
    // Linear index into the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = batch_size * channels * pooled_height * pooled_width;
    if (idx >= total_count) return;

    // Decode idx into (b, c, ph, pw)
    int pw = idx % pooled_width;
    int tmp = idx / pooled_width;
    int ph = tmp % pooled_height;
    tmp = tmp / pooled_height;
    int c = tmp % channels;
    int b = tmp / channels;

    // Compute the start/end for the pooling window
    int h_start = ph * stride - padding;
    int w_start = pw * stride - padding;
    // The effective kernel size in each dimension
    int h_end = h_start + kernel_size * dilation;
    int w_end = w_start + kernel_size * dilation;

    float max_val = -FLT_MAX;

    // Iterate over the pooling window
    for (int h = h_start; h < h_end; h += dilation) {
        for (int w = w_start; w < w_end; w += dilation) {
            if (h >= 0 && h < height && w >= 0 && w < width) {
                int in_idx = ((b * channels + c) * height + h) * width + w;
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    // Write result to output
    output[idx] = max_val;
}

// Forward launcher for 2D max pooling
torch::Tensor max_pool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels   = sizes[1];
    int height     = sizes[2];
    int width      = sizes[3];

    // Calculate output dimensions
    int pooled_height = (height + 2 * padding - (kernel_size - 1) * dilation - 1) / stride + 1;
    int pooled_width  = (width  + 2 * padding - (kernel_size - 1) * dilation - 1) / stride + 1;

    auto options = input.options();
    torch::Tensor output = torch::empty({batch_size, channels, pooled_height, pooled_width}, options);

    int total_count = batch_size * channels * pooled_height * pooled_width;

    // Configure grid/block
    int threads = 256;
    int blocks = (total_count + threads - 1) / threads;

    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        pooled_height,
        pooled_width
    );

    return output;
}

// Pybind11 binding 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &max_pool2d_forward,
        "MaxPool2D forward (CUDA)"
    );
}
