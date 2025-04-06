#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void cumsum_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int outer_size, int inner_size, int inner_block) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_lines = outer_size * inner_block;
  if (tid >= num_lines) return;
  int line_idx = tid / inner_block;
  int block_idx = tid % inner_block;
  int base = line_idx * inner_size * inner_block + block_idx;
  scalar_t sum = 0;
  for (int i = 0; i < inner_size; ++i) {
    int idx = base + i * inner_block;
    sum += input[idx];
    output[idx] = sum;
  }
}

torch::Tensor cumsum_cuda(torch::Tensor input, int64_t dim) {
  auto x = input.contiguous();
  auto sizes = x.sizes();
  int D = x.dim();
  int inner_block = 1;
  for (int i = D - 1; i > dim; --i) {
    inner_block *= sizes[i];
  }
  int inner_size = sizes[dim];
  int outer_size = x.numel() / (inner_size * inner_block);
  auto output = torch::empty_like(x);
  int num_lines = outer_size * inner_block;
  const int threads = 512;
  const int blocks = (num_lines + threads - 1) / threads;
  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "cumsum_cuda", ([&] {
    cumsum_kernel<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      outer_size, inner_size, inner_block);
  }));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cumsum_cuda, "Cumulative sum forward");
}