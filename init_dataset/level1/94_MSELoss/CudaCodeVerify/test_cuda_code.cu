#include <torch/extension.h>

__global__ void mse_loss_kernel(const float *predictions, const float *targets, float *loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        loss[idx] = diff * diff;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    auto loss = torch::zeros_like(predictions);
    int n = predictions.numel();

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), loss.data_ptr<float>(), n);
    
    cudaDeviceSynchronize();
    
    return loss.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error Loss forward function");
}