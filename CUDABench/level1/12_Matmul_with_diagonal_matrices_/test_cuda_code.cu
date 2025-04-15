#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void diag_matmul_kernel(const scalar_t* __restrict__ A,
                                   const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C,
                                   int N, int M) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < M) {
    C[row * M + col] = A[row] * B[row * M + col];
  }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
  const int N = A.size(0);
  const int M = B.size(1);
  auto C = torch::empty({N, M}, B.options());

  const int threads = 16;
  const dim3 threadsPerBlock(threads, threads);
  const dim3 blocks((M + threads - 1) / threads, (N + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "diag_matmul_cuda", ([&] {
    diag_matmul_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
      A.data_ptr<scalar_t>(),
      B.data_ptr<scalar_t>(),
      C.data_ptr<scalar_t>(),
      N,
      M
    );
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &diag_matmul_cuda, "Diagonal Matrix Multiplication (CUDA)");
}