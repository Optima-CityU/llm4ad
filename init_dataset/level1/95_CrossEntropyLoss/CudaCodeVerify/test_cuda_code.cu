// cross_entropy_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float log_sum_exp(float* logits, int num_classes) {
    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, logits[i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum_exp += expf(logits[i] - max_val);
    }
    
    return logf(sum_exp) + max_val;
}

__global__ void cross_entropy_kernel(float* predictions, int* targets, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int target_class = targets[idx];
        float* logits = predictions + idx * num_classes;
        
        float log_sum_exp_val = log_sum_exp(logits, num_classes);
        float log_pred = logits[target_class] - log_sum_exp_val;
        
        atomicAdd(loss, -log_pred);
    }
}

torch::Tensor cross_entropy_forward_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    float* predictions_ptr = predictions.data_ptr<float>();
    int* targets_ptr = targets.data_ptr<int>();
    
    torch::Tensor loss = torch::zeros({1}, predictions.options());
    float* loss_ptr = loss.data_ptr<float>();

    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    cross_entropy_kernel<<<blocks, threads_per_block>>>(predictions_ptr, targets_ptr, loss_ptr, batch_size, num_classes);

    cudaDeviceSynchronize();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cross_entropy_forward_cuda, "Cross Entropy Loss (CUDA)");
}