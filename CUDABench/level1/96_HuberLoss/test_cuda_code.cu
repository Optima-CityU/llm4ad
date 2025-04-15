#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ losses,
    int64_t numel,
    float beta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float x = predictions[idx];
    float y = targets[idx];
    float diff = x - y;
    float abs_diff = fabsf(diff);
    if (abs_diff < beta) {
      losses[idx] = 0.5f * diff * diff / beta;
    } else {
      losses[idx] = abs_diff - 0.5f * beta;
    }
  }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets) {
  CHECK_INPUT(predictions);
  CHECK_INPUT(targets);

  const auto numel = predictions.numel();
  const float beta = 1.0f;

  auto losses = torch::empty_like(predictions);

  const int threads = 512;
  const int blocks = (numel + threads - 1) / threads;

  smooth_l1_loss_kernel<<<blocks, threads>>>(
      predictions.data_ptr<float>(),
      targets.data_ptr<float>(),
      losses.data_ptr<float>(),
      numel,
      beta);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

  auto loss = losses.mean();

  return loss;
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
  return smooth_l1_loss_cuda(predictions, targets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Smooth L1 Loss forward (CUDA)");
}