#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w,
    int groups) {

    int n  = blockIdx.x;
    int oc = blockIdx.y;
    int idx = blockIdx.z;
    int oh = idx / width_out;
    int ow = idx % width_out;
    if (n >= batch_size || oc >= out_channels || oh >= height_out || ow >= width_out) return;

    float value = 0.0f;
    int in_per_group  = in_channels  / groups;
    int out_per_group = out_channels / groups;
    int group_id      = oc / out_per_group;
    int ic_start      = group_id * in_per_group;
    int ic_end        = ic_start + in_per_group;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        int ic_local = ic - ic_start;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh * dil_h;
                int iw = ow * stride_w - pad_w + kw * dil_w;
                if (ih >= 0 && ih < height_in && iw >= 0 && iw < width_in) {
                    int inp_idx = ((n * in_channels + ic) * height_in + ih) * width_in + iw;
                    int w_idx   = ((oc * in_per_group + ic_local) * kernel_h + kh) * kernel_w + kw;
                    value += input[inp_idx] * weight[w_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        value += bias[oc];
    }

    int out_idx = ((n * out_channels + oc) * height_out + oh) * width_out + ow;
    output[out_idx] = value;
}

torch::Tensor conv2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias_opt.has_value()) {
        CHECK_INPUT(bias_opt.value());
    }

    auto batch_size   = input.size(0);
    auto in_channels  = input.size(1);
    auto height_in    = input.size(2);
    auto width_in     = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h     = weight.size(2);
    auto kernel_w     = weight.size(3);

    int64_t stride_h = stride[0], stride_w = stride[1];
    int64_t pad_h    = padding[0], pad_w    = padding[1];
    int64_t dil_h    = dilation[0], dil_w   = dilation[1];

    auto height_out = (height_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto width_out  = (width_in  + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const float* bias_ptr   = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;
    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float*       output_ptr = output.data_ptr<float>();

    dim3 grid(batch_size, out_channels, height_out * width_out);
    dim3 block(1);

    conv2d_cuda_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        height_in, width_in, height_out, width_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_forward, "2D Convolution forward (CUDA)");
}