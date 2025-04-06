// Include the necessary headers
#include <torch/extension.h>
#include <cmath>

// CUDA kernel for Smooth L1 loss
__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* output, const int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        float diff = predictions[index] - targets[index];
        float abs_diff = fabs(diff);

        // Smooth L1 loss (Huber loss)
        if (abs_diff < 1.0f) {
            output[index] = 0.5f * diff * diff;
        } else {
            output[index] = abs_diff - 0.5f;
        }
    }
}

// Function to launch the kernel and compute Smooth L1 loss
at::Tensor smooth_l1_loss_cuda(at::Tensor predictions, at::Tensor targets) {
    int N = predictions.numel();

    // Ensure the tensors are the same size
    assert(predictions.sizes() == targets.sizes());

    // Allocate output tensor
    at::Tensor output = at::zeros_like(predictions);

    // Set up the grid and block sizes
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    smooth_l1_loss_kernel<<<blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    // Synchronize the device
    cudaDeviceSynchronize();

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smooth_l1_loss", &smooth_l1_loss_cuda, "Smooth L1 Loss (Huber Loss) CUDA");
}