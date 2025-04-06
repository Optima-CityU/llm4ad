#include <torch/extension.h>

__global__ void max_pool1d_kernel(
    const float *input, 
    float *output, 
    int *indices, 
    int batch_size, 
    int channels, 
    int length, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation, 
    bool return_indices) 
{
    int b = blockIdx.x;  // batch index
    int c = blockIdx.y;  // channel index
    int i = threadIdx.x + blockIdx.z * blockDim.x;  // input index

    if (i < length) {
        int start = max(i * stride - padding, 0);
        int end = min(i * stride - padding + kernel_size * dilation, length);
        int max_val_idx = -1;
        float max_val = -FLT_MAX;
        
        for (int j = start; j < end; j += dilation) {
            float value = input[(b * channels + c) * length + j];
            if (value > max_val) {
                max_val = value;
                max_val_idx = j;
            }
        }

        output[(b * channels + c) * length + i] = max_val;

        if (return_indices) {
            indices[(b * channels + c) * length + i] = max_val_idx;
        }
    }
}

torch::Tensor max_pool1d_forward(
    torch::Tensor input, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation, 
    bool return_indices) 
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int length = input.size(2);

    auto options = input.options().dtype(torch::kFloat32);
    auto output = torch::empty({batch_size, channels, length}, options);
    auto indices = return_indices ? torch::empty({batch_size, channels, length}, torch::kInt32) : torch::empty({0}, torch::kInt32);

    const int block_size = 256;
    const int num_blocks = (length + block_size - 1) / block_size;
    
    max_pool1d_kernel<<<dim3(batch_size, channels, num_blocks), block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        indices.data_ptr<int>(), 
        batch_size, 
        channels, 
        length, 
        kernel_size, 
        stride, 
        padding, 
        dilation, 
        return_indices
    );

    return return_indices ? std::make_tuple(output, indices) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_forward", &max_pool1d_forward, "Max pool 1D forward");
}