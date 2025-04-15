#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int W_in,
    const int kernel_size, const int stride, const int padding,
    const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * W_out) return;

    int n = idx / (C * W_out);
    int c = (idx / W_out) % C;
    int w_out = idx % W_out;

    float sum_val = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int w_in = w_out * stride - padding + k;
        if (w_in >= 0 && w_in < W_in) {
            sum_val += input[n * C * W_in + c * W_in + w_in];
        }
    }
    output[idx] = sum_val / static_cast<float>(kernel_size);
}

torch::Tensor forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding)
{
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto W_in = input.size(2);

    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({N, C, W_out}, input.options());

    const int threads = 256;
    const int blocks = (N * C * W_out + threads - 1) / threads;

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, W_in,
        kernel_size, stride, padding,
        W_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D average pooling forward (CUDA)");
}