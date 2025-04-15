#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    scalar_t val = input[idx];
    output[idx] = (val > 0) ? val : static_cast<scalar_t>(0);
  }
}

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_forward_cuda", ([&] {
    relu_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                 output.data_ptr<scalar_t>(),
                                                 numel);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "ReLU forward (CUDA)");
}