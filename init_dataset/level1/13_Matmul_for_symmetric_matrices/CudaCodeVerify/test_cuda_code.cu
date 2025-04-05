#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
  // Check that inputs are CUDA tensors and are square matrices
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
  TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1),
              "Input matrices must be square");

  int N = A.size(0);
  auto C = torch::zeros({N, N}, A.options());

  // Define block and grid dimensions
  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

  // Launch the CUDA kernel
  matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
  cudaDeviceSynchronize();

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Matrix multiplication of symmetric matrices (CUDA)");
}