#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

// CUDA kernel to compute softmax along dimension 1 for each row.
__global__ void softmax_kernel(const float* input, float* output, int num_features) {
    // Each block processes one row.
    int row = blockIdx.x;
    // Pointers to the start of the row.
    input  += row * num_features;
    output += row * num_features;
    
    // Use shared memory for reduction.
    extern __shared__ float sdata[];

    // Step 1: Compute the maximum value in the row for numerical stability.
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_features; j += blockDim.x) {
        thread_max = fmaxf(thread_max, input[j]);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduction to find the maximum value.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Step 2: Compute the sum of exp(x - max_val).
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < num_features; j += blockDim.x) {
        thread_sum += expf(input[j] - max_val);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction to compute the total sum.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Step 3: Compute the softmax output.
    for (int j = threadIdx.x; j < num_features; j += blockDim.x) {
        output[j] = expf(input[j] - max_val) / sum_exp;
    }
}

// C++ interface for PyTorch.
torch::Tensor softmax_forward(torch::Tensor input) {
    // Allocate output tensor with the same size and options as input.
    auto output = torch::empty_like(input);

    int batch_size = input.size(0);
    int num_features = input.size(1);

    // Choose number of threads per block (capped at 256).
    int threads = (num_features < 256) ? num_features : 256;
    int blocks = batch_size;

    // Launch the CUDA kernel with dynamic shared memory for reduction.
    softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Softmax forward (CUDA)");
}