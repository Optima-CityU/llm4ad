// Includes
#include <torch/extension.h>

// CUDA kernel for summing along a specified dimension
__global__ void sum_reduction_kernel(const float* input, float* output, int dim, int batch_size, int reduced_dim, int other_dim_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batch_size * other_dim_size) {
        int batch_idx = index / other_dim_size;
        int other_idx = index % other_dim_size;
        
        float sum = 0.0f;
        for (int i = 0; i < reduced_dim; i++) {
            int input_idx = (dim == 0) 
                             ? (i * other_dim_size + other_idx + batch_idx * reduced_dim * other_dim_size) 
                             : (batch_idx * reduced_dim * other_dim_size + i * other_dim_size + other_idx);
            sum += input[input_idx];
        }

        int output_idx = batch_idx * 1 * other_dim_size + other_idx;
        output[output_idx] = sum;
    }
}

// Wrapper function to launch the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    // Get dimensions
    auto sizes = x.sizes();
    int batch_size = sizes[0]; 
    int reduced_dim = sizes[dim]; 
    int other_dim_size = sizes[dim == 0 ? 1 : 2]; 

    // Allocate output tensor
    auto output = torch::zeros({batch_size, 1, other_dim_size}, x.options());

    // Launch kernel
    int total_elements = batch_size * other_dim_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    sum_reduction_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        dim, 
        batch_size, 
        reduced_dim, 
        other_dim_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA sum reduction along specified dimension");
}