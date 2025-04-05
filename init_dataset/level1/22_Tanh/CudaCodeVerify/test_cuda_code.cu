
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_kernel(const float* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int size = input.numel();
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  tanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Apply Tanh activation (CUDA)");
}
