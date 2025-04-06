// hinge_loss_kernel.cu
#include <torch/extension.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float pred = predictions[idx];
        float target = targets[idx];
        float diff = 1.0f - pred * target;
        loss[idx] = (diff > 0.0f) ? diff : 0.0f;
    }
}

torch::Tensor hinge_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    
    // Allocate memory for the loss
    torch::Tensor loss = torch::zeros({batch_size}, predictions.options());

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    hinge_loss_kernel<<<num_blocks, threads_per_block>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), loss.data_ptr<float>(), batch_size);

    // Compute the mean of the loss
    float total_loss = loss.sum().item<float>();
    return torch::tensor(total_loss / batch_size, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hinge_loss_forward, "Hinge loss computation (CUDA)");
}