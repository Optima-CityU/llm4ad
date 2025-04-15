#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

// Kernel to compute partial sums of squares for reduction
__global__ void sum_squares_kernel(const float* __restrict__ input, float* __restrict__ block_sum, int n) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // Grid-stride loop to accumulate squares
    while (i < n) {
        float val = input[i];
        sum += val * val;
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    // Intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write result of this block to global memory
    if (tid == 0) {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// Kernel to normalize the input tensor using the computed norm
__global__ void normalize_kernel(const float* __restrict__ input, float* __restrict__ output, float norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] / norm;
    }
}

torch::Tensor frobenius_norm_forward(torch::Tensor x) {
    // Ensure the input is contiguous and on CUDA
    auto input = x.contiguous();
    int n = input.numel();
    auto output = torch::empty_like(input);

    // Get raw pointers to the data
    const float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    // First reduction step: compute sum of squares using the input tensor.
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;
    // Allocate temporary tensor for block sums
    auto block_sums = torch::empty({blocks}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
    float* d_block_sums = block_sums.data_ptr<float>();

    sum_squares_kernel<<<blocks, threads>>>(d_input, d_block_sums, n);

    // Perform iterative reduction if necessary
    int s = blocks;
    while (s > 1) {
        int threads2 = THREADS_PER_BLOCK;
        int blocks2 = (s + threads2 - 1) / threads2;
        auto temp = torch::empty({blocks2}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
        float* d_temp = temp.data_ptr<float>();
        sum_squares_kernel<<<blocks2, threads2>>>(d_block_sums, d_temp, s);
        s = blocks2;
        block_sums = temp;
        d_block_sums = block_sums.data_ptr<float>();
    }

    // Copy the final sum of squares from device to host
    float sum_squares;
    cudaMemcpy(&sum_squares, d_block_sums, sizeof(float), cudaMemcpyDeviceToHost);
    float norm = sqrtf(sum_squares);

    // Launch kernel to normalize the tensor
    int norm_threads = THREADS_PER_BLOCK;
    int norm_blocks = (n + norm_threads - 1) / norm_threads;
    normalize_kernel<<<norm_blocks, norm_threads>>>(d_input, d_output, norm, n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &frobenius_norm_forward, "Frobenius Norm Normalization (CUDA)");
}