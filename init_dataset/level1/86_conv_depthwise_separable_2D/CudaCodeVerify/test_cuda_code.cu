#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors (optional)
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

// Depthwise convolution kernel
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C,
    const int H, const int W,
    const int KH, const int KW,
    const int stride, const int padding, const int dilation,
    const int H_out, const int W_out) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (index < total) {
        // Compute indices for n, c, output height and width
        int ow = index % W_out;
        int tmp = index / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        float sum = 0.f;
        // Loop over kernel height and width
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = oh * stride - padding + kh * dilation;
                int w_in = ow * stride - padding + kw * dilation;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_index = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                    int weight_index = c * (KH * KW) + kh * KW + kw;
                    sum += input[in_index] * weight[weight_index];
                }
            }
        }
        sum += bias[c];
        output[index] = sum;
    }
}

// Pointwise convolution kernel
__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C, // C: number of channels from depthwise output
    const int H_out, const int W_out,
    const int out_channels) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * H_out * W_out;
    if (index < total) {
        int ow = index % W_out;
        int tmp = index / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        float sum = 0.f;
        // Sum over the input channels
        for (int c = 0; c < C; ++c) {
            int in_index = n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out + ow;
            int weight_index = oc * C + c; // pointwise kernel is 1x1
            sum += input[in_index] * weight[weight_index];
        }
        sum += bias[oc];
        output[index] = sum;
    }
}

// Forward function: performs depthwise separable convolution
torch::Tensor forward(
    torch::Tensor x, 
    torch::Tensor depthwise_weight, 
    torch::Tensor depthwise_bias, 
    torch::Tensor pointwise_weight, 
    torch::Tensor pointwise_bias, 
    int stride, int padding, int dilation) 
{
    CHECK_CUDA(x);
    CHECK_CUDA(depthwise_weight);
    CHECK_CUDA(depthwise_bias);
    CHECK_CUDA(pointwise_weight);
    CHECK_CUDA(pointwise_bias);

    // Input dimensions
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    // Depthwise kernel dimensions (assumes weight shape: [C, 1, KH, KW])
    const int KH = depthwise_weight.size(2);
    const int KW = depthwise_weight.size(3);

    // Calculate output spatial dimensions for depthwise convolution
    const int H_out = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    const int W_out = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;

    // Allocate intermediate tensor for depthwise output
    auto depthwise_out = torch::empty({N, C, H_out, W_out}, x.options());

    // Launch depthwise convolution kernel
    int total_depth = N * C * H_out * W_out;
    int threads = 1024;
    int blocks = (total_depth + threads - 1) / threads;
    depthwise_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        depthwise_bias.data_ptr<float>(),
        depthwise_out.data_ptr<float>(),
        N, C, H, W, KH, KW, stride, padding, dilation, H_out, W_out);
    cudaDeviceSynchronize();

    // Pointwise convolution: weight shape assumed [out_channels, C] for 1x1 conv
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({N, out_channels, H_out, W_out}, x.options());

    int total_point = N * out_channels * H_out * W_out;
    blocks = (total_point + threads - 1) / threads;
    pointwise_conv2d_kernel<<<blocks, threads>>>(
        depthwise_out.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        pointwise_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_out, W_out, out_channels);
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise-Separable Convolution forward (CUDA)");
}