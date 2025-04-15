// Necessary pybind11 and CUDA module imports
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float *input, float *output, int num_elements, float min_val, float max_val) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        float val = input[index];
        if (val < min_val) {
            output[index] = min_val;
        } else if (val > max_val) {
            output[index] = max_val;
        } else {
            output[index] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    int num_elements = input.numel();
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel<<<blocks, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements, -1.0f, 1.0f);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hardtanh_cuda, "HardTanh activation (CUDA)");
}