#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel: compute element-wise contribution to KL divergence
__global__ void kl_div_kernel(const float* predictions, const float* targets, int total_elements, float* result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_elements) {
    float pred = predictions[i];
    float target = targets[i];
    float term = 0.0f;
    // Only compute the term if target > 0, as defined in KL divergence
    if (target > 0.0f) {
      term = target * (logf(target) - logf(pred));
    }
    atomicAdd(result, term);
  }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
  // Ensure the tensors are on CUDA and have the same shape.
  TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same size");

  int total_elements = predictions.numel();
  // 'batchmean' reduction divides by the batch size (assumed to be the size of dim 0)
  int batch_size = predictions.size(0);

  auto options = torch::TensorOptions().dtype(predictions.dtype()).device(predictions.device());
  auto result_tensor = torch::zeros({1}, options);

  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  // Launch the CUDA kernel.
  kl_div_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(),
                                       targets.data_ptr<float>(),
                                       total_elements,
                                       result_tensor.data_ptr<float>());
  cudaDeviceSynchronize();

  // Apply the 'batchmean' reduction: divide the sum by the batch size.
  float result_value = result_tensor.item<float>() / batch_size;

  return torch::tensor(result_value, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KL Divergence forward (CUDA)");
}