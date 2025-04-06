// product_reduction_kernel.cu
#include <torch/extension.h>

__global__ void product_reduction_kernel(const float* x, float* out, int64_t N, int64_t dim, int64_t size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        float prod = 1.0f;
        
        // Perform the reduction over the dimension `dim`
        for (int i = 0; i < size; ++i) {
            prod *= x[index * size + i];
        }
        
        out[index] = prod;
    }
}

torch::Tensor product_reduction(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    // Get the size of the specified dimension
    int64_t N = x.size(0);
    int64_t size = x.size(dim);
    auto out = torch::empty({N, size}, x.options());

    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    product_reduction_kernel<<<blocks, threadsPerBlock>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, dim, size);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &product_reduction, "Product reduction over dimension (CUDA)");
}