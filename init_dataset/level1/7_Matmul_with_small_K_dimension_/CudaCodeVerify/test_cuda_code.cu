#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
  const auto M = A.size(0);
  const auto K = A.size(1);
  const auto N = B.size(1);

  auto C = torch::zeros({M, N}, A.options());

  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

  matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul, "Matrix multiplication (CUDA)");
}