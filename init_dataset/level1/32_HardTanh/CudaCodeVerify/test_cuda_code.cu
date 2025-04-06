// Includes necessary headers
#include <torch/extension.h>

__global__ void hardtanh_kernel(float* input, float* output, int64_t num_elements, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float value = input[idx];
        if (value < min_val) {
            output[idx] = min_val;
        } else if (value > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = value;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    
    // Define block and grid sizes
    dim3 threads(256);
    dim3 blocks((num_elements + threads.x - 1) / threads.x);
    
    // Launch kernel
    hardtanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements, min_val, max_val);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardtanh_cuda", &hardtanh_cuda, "HardTanh activation (CUDA)");
}