|
  #include <torch/extension.h>
  #include <cuda.h>
  #include <cuda_runtime.h>

  #define TILE_WIDTH 32
  #define VECTOR_WIDTH 4

  // This kernel uses CUDA 12.4’s asynchronous copy (cp.async) instructions available on H100
  // to prefetch tiles into shared memory. It keeps the original 4‐way unrolling and double buffering
  // strategy, but replaces the blocking loads with asynchronous copies to overlap memory transfer with computation.
  template <typename scalar_t>
  __global__ void compute_async_shared_kernel(const scalar_t* __restrict__ A,
                                                const scalar_t* __restrict__ B,
                                                scalar_t* __restrict__ C,
                                                int N) {
      // Allocate two ping–pong shared memory buffers for A and B.
      __shared__ scalar_t sA[2][TILE_WIDTH][TILE_WIDTH + 2];
      __shared__ scalar_t sB[2][TILE_WIDTH][TILE_WIDTH + 2];

      const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
      const int col = blockIdx.x * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
      scalar_t value[VECTOR_WIDTH] = {0};

      const int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
      int stage = 0;

      // Preload the first tile asynchronously using cp.async.
      int tile = 0;
      int a_col = tile * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
      if (row < N && (a_col + VECTOR_WIDTH - 1) < N) {
          scalar_t* dest = &sA[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
          const scalar_t* src = &A[row * N + a_col];
          asm volatile(
              "cp.async.cg.shared.global [%0], [%1], %2;\n"
              :
              : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
              : "memory");
      } else {
          // Fallback for out‐of‐range elements.
          scalar_t* dest = &sA[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
          for (int i = 0; i < VECTOR_WIDTH; i++) {
              int idx = a_col + i;
              dest[i] = (row < N && idx < N) ? A[row * N + idx] : 0;
          }
      }

      int b_row = tile * TILE_WIDTH + threadIdx.y;
      if (b_row < N && (col + VECTOR_WIDTH - 1) < N) {
          scalar_t* dest = &sB[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
          const scalar_t* src = &B[b_row * N + col];
          asm volatile(
              "cp.async.cg.shared.global [%0], [%1], %2;\n"
              :
              : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
              : "memory");
      } else {
          scalar_t* dest = &sB[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
          for (int i = 0; i < VECTOR_WIDTH; i++) {
              int idx = col + i;
              dest[i] = (b_row < N && idx < N) ? B[b_row * N + idx] : 0;
          }
      }
      // Wait for asynchronous copies to complete.
      asm volatile("cp.async.wait_all;\n" : : : "memory");
      __syncthreads();

      // Main loop over tiles with double buffering.
      for (int t = 0; t < num_tiles; t++) {
          int next_stage = 1 - stage;
          if (t + 1 < num_tiles) {
              int next_tile = t + 1;
              int a_col_next = next_tile * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
              if (row < N && (a_col_next + VECTOR_WIDTH - 1) < N) {
                  scalar_t* dest = &sA[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
                  const scalar_t* src = &A[row * N + a_col_next];
                  asm volatile(
                      "cp.async.cg.shared.global [%0], [%1], %2;\n"
                      :
                      : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
                      : "memory");
              } else {
                  scalar_t* dest = &sA[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
                  for (int i = 0; i < VECTOR_WIDTH; i++) {
                      int idx = a_col_next + i;
                      dest[i] = (row < N && idx < N) ? A[row * N + idx] : 0;
                  }
              }

              int b_row_next = next_tile * TILE_WIDTH + threadIdx.y;
              if (b_row_next < N && (col + VECTOR_WIDTH - 1) < N) {
                  scalar_t* dest = &sB[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
                  const scalar_t* src = &B[b_row_next * N + col];
                  asm volatile(
                      "cp.async.cg.shared.global [%0], [%1], %2;\n"
                      :
                      : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
                      : "memory");
              } else {
                  scalar_t* dest = &sB[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
                  for (int i = 0; i < VECTOR_WIDTH; i++) {
                      int idx = col + i;
                      dest[i] = (b_row_next < N && idx < N) ? B[b_row_next * N + idx] : 0;
                  }
              }
          }

          asm volatile("cp.async.wait_all;\n" : : : "memory");
          __syncthreads();

          // Compute the current tile using 4-way unrolling.
          #pragma unroll
          for (int k = 0; k < TILE_WIDTH; k += 4) {
              scalar_t a0 = sA[stage][threadIdx.y][k];
              scalar_t a1 = sA[stage][threadIdx.y][k + 1];
              scalar_t a2 = sA[stage][threadIdx.y][k + 2];
              scalar_t a3 = sA[stage][threadIdx.y][k + 3];

              scalar_t b0[VECTOR_WIDTH], b1[VECTOR_WIDTH], b2[VECTOR_WIDTH], b3[VECTOR_WIDTH];
              for (int v = 0; v < VECTOR_WIDTH; ++v) {
                  b0[v] = sB[stage][k][threadIdx.x * VECTOR_WIDTH + v];
                  b1[v] = sB[stage][k + 1][threadIdx.x * VECTOR_WIDTH + v];
                  b2[v] = sB[stage][k + 2][threadIdx.x * VECTOR_WIDTH + v];
                  b3[v] = sB[stage][k + 3][threadIdx.x * VECTOR_WIDTH + v];
              }

              for (int v = 0; v < VECTOR_WIDTH; ++v) {
                  value[v] = __fmaf_rn(a0, b0[v], value[v]);
                  value[v] = __fmaf_rn(a1, b1[v], value[v]);
                  value[v] = __fmaf_rn(a2, b2[v], value[v]);
                  value[v] = __fmaf_rn(a3, b3[v], value[v]);
              }
          }

          __syncthreads();
          stage = next_stage;
      }

      // Write back the computed tile to global memory.
      if (row < N && col < N) {
          if (col + VECTOR_WIDTH <= N) {
              float4 out_val = {value[0], value[1], value[2], value[3]};
              *reinterpret_cast<float4*>(&C[row * N + col]) = out_val;
          } else {
              for (int v = 0; v < VECTOR_WIDTH; ++v) {
                  if (col + v < N)
                      C[row * N + col + v] = value[v];
              }
          }
      }
  }

  torch::Tensor compute_async_shared_matmul_cuda(torch::Tensor A, torch::Tensor B) {
      const int N = A.size(0);
      A = A.contiguous();
      B = B.contiguous();
      auto C = torch::zeros({N, N}, A.options());

      const int threads_x = TILE_WIDTH / VECTOR_WIDTH;
      dim3 block_dim(threads_x, TILE_WIDTH);
      dim3 grid_dim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

      AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "compute_async_shared_matmul_cuda", ([&] {
          compute_async_shared_kernel<scalar_t><<<grid_dim, block_dim>>>(
              A.data_ptr<scalar_t>(),
              B.data_ptr<scalar_t>(),
              C.data_ptr<scalar_t>(),
              N);
      }));

      return C;
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &compute_async_shared_matmul_cuda, "Async Shared Memory Matrix Multiplication CUDA forward");
  }