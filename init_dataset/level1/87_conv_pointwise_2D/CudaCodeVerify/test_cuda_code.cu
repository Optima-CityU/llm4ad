#include <torch/extension.h>

__global__ void conv2d_kernel(const float* x, const float* weight, const float* bias, float* out, 
                               int batch_size, int in_channels, int height, int width, 
                               int out_channels) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < batch_size && m < out_channels) {
        int out_h = blockIdx.z % height;
        int out_w = blockIdx.z / height;

        float sum = 0.0f;
        for (int c = 0; c < in_channels; c++) {
            int idx_in = (n * in_channels + c) * height * width + out_h * width + out_w;
            int idx_weight = (m * in_channels + c) * 1 * 1;
            sum += x[idx_in] * weight[idx_weight];
        }

        out[(n * out_channels + m) * height * width + out_h * width + out_w] = sum + bias[m];
    }
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);
    const int out_channels = weight.size(0);

    torch::Tensor out = torch::zeros({batch_size, out_channels, height, width}, x.options());

    dim3 threads(16, 16);
    dim3 blocks((batch_size + threads.x - 1) / threads.x, 
                (out_channels + threads.y - 1) / threads.y,
                height * width);

    conv2d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), 
                                       bias.data_ptr<float>(), out.data_ptr<float>(), 
                                       batch_size, in_channels, height, width, out_channels);

    return out;
}