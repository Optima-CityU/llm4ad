
```C++
// Includes for CUDA and PyTorch
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* x, const float* gn_weight, const float* gn_bias, 
                                   float* output, int batch_size, int num_features, 
                                   int num_groups, int group_size, int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < num_elements) {
        int feature_idx = index % num_features;
        int batch_idx = index / num_features;

        // Calculate the group index and within group index
        int group_idx = feature_idx / group_size;
        int group_offset = group_idx * group_size;

        // Calculate mean and variance for normalization
        float mean = 0.0f, variance = 0.0f;
        for (int i = 0; i < group_size; i++) {
            int group_feature_idx = group_offset + i;
            mean += x[batch_idx * num_features + group_feature_idx];
        }
        mean /= group_size;

        for (int i = 0; i < group_size; i++) {
            int group_feature_idx = group_offset + i;
            variance += (x[batch_idx * num_features + group_feature_idx] - mean) *
                        (x[batch_idx * num_features + group_feature_idx] - mean);
        }
        variance /= group_size;
        float stddev = sqrtf(variance + 1e-5f);

        // Apply group normalization
        output[batch_idx * num_features + feature_idx] = 
            gn_weight[feature_idx] * ((x[batch_idx * num_features + feature_idx] - mean) / stddev) + gn_bias[feature_idx];
    }
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor gn_weight, torch::Tensor gn_bias, int num_groups) {
    int batch_size = x.size(0);
    int num_features = x.size(1);
    int group_size = num_features / num_groups;
    int num_elements = batch_size * num_features;

    // Allocate memory for output tensor
    torch::Tensor output = torch::empty_like(x);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    group_norm_kernel<<<num_blocks, threads_per_block>>>(x.data_ptr<float>(), gn_weight.data_ptr<float>(), 
                                                          gn_bias.data_ptr<float>(), output.data_ptr<float>(), 
                                                          batch_size, num_features, num_groups, group_size, num_elements);

    // Return the result
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_cuda, "Group Normalization (CUDA)");
}
