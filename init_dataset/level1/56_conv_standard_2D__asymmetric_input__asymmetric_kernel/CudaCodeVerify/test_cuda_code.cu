#include <torch/extension.h>
#include <vector>

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int in_channels_per_group)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (index >= total) return;

    // Calculate output coordinates
    int w_out = index % out_w;
    int h_out = (index / out_w) % out_h;
    int oc = (index / (out_w * out_h)) % out_channels;
    int b = index / (out_w * out_h * out_channels);

    int group = oc / (out_channels / groups);
    int in_ch_start = group * in_channels_per_group;

    float value = bias[oc];
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        int actual_ic = in_ch_start + ic;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    int input_idx = ((b * in_channels + actual_ic) * in_h + h_in) * in_w + w_in;
                    int weight_idx = (((oc * in_channels_per_group) + ic) * kernel_h + kh) * kernel_w + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    int output_idx = ((b * out_channels + oc) * out_h + h_out) * out_w + w_out;
    output[output_idx] = value;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups)
{
    // Check tensors are on CUDA
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    // Get input dimensions
    const auto batch = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_h = x.size(2);
    const auto in_w = x.size(3);

    // Get weight dimensions
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Stride, padding, dilation
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    // Calculate output dimensions
    const auto out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const auto out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch, out_channels, out_h, out_w}, x.options());

    // Determine the number of input channels per group
    int in_channels_per_group = in_channels / groups;

    // Total number of output elements
    int total_elements = batch * out_channels * out_h * out_w;

    // Launch CUDA kernel
    const int threads = 1024;
    const int blocks = (total_elements + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels_per_group
    );

    // Ensure kernel launch errors are reported
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Convolution forward (CUDA)");
}