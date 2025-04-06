// CUDA kernel for transposed 1D convolution
#include <torch/extension.h>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int length_in,
    const int length_out,
    const int stride,
    const int padding,
    const int dilation
) {
    int batch_idx = blockIdx.x;
    int out_channel_idx = blockIdx.y;
    int out_pos = threadIdx.x;

    if (out_pos >= length_out) return;

    int in_pos_start = out_pos * stride - padding;
    for (int in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
        for (int kernel_pos = 0; kernel_pos < weight.size(2); kernel_pos++) {
            int in_pos = in_pos_start + kernel_pos * dilation;

            if (in_pos >= 0 && in_pos < length_in) {
                float input_val = input[batch_idx * in_channels * length_in + in_channel_idx * length_in + in_pos];
                float weight_val = weight[out_channel_idx * in_channels * weight.size(2) + in_channel_idx * weight.size(2) + kernel_pos];
                output[batch_idx * out_channels * length_out + out_channel_idx * length_out + out_pos] += input_val * weight_val;
            }
        }
    }

    if (bias != nullptr) {
        output[batch_idx * out_channels * length_out + out_channel_idx * length_out + out_pos] += bias[out_channel_idx];
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int length_in = input.size(2);
    const int length_out = (length_in - 1) * stride - 2 * padding + dilation * (weight.size(2) - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, length_out}, input.options());

    const dim3 block_size(length_out);
    const dim3 grid_size(batch_size, out_channels);

    conv_transpose1d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        length_in,
        length_out,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose1d_cuda", &conv_transpose1d_cuda, "Transposed 1D convolution (CUDA)");
}