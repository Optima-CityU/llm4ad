#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void triplet_loss_opt_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    const int D,
    const scalar_t margin,
    scalar_t* loss_array) {

  // Shared memory to accumulate warp‐level sums.
  __shared__ scalar_t warp_sdata_ap[BLOCK_SIZE / 32];
  __shared__ scalar_t warp_sdata_an[BLOCK_SIZE / 32];

  int sample = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  const unsigned int mask = 0xffffffff;

  scalar_t partial_ap = 0;
  scalar_t partial_an = 0;

  // Base pointers for current sample.
  const scalar_t* a_ptr = anchor + sample * D;
  const scalar_t* p_ptr = positive + sample * D;
  const scalar_t* n_ptr = negative + sample * D;

  // Use vectorized loads if possible.
  // For float, use groups of 4; for double, use groups of 2.
  int remainder = D;
  if (sizeof(scalar_t) == 4) {
    const int vec_size = 4;
    int numVec = D / vec_size;
    remainder = D % vec_size;
    // Each thread processes several vectorized loads.
    for (int i = tid; i < numVec; i += blockDim.x) {
      // reinterpret as float4.
      const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
      const float4* p_vec = reinterpret_cast<const float4*>(p_ptr);
      const float4* n_vec = reinterpret_cast<const float4*>(n_ptr);
      float4 a_val = __ldg(&a_vec[i]);
      float4 p_val = __ldg(&p_vec[i]);
      float4 n_val = __ldg(&n_vec[i]);
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        float a_elem = ((float*)&a_val)[j];
        float p_elem = ((float*)&p_val)[j];
        float n_elem = ((float*)&n_val)[j];
        float diff_ap = a_elem - p_elem;
        float diff_an = a_elem - n_elem;
        partial_ap += diff_ap * diff_ap;
        partial_an += diff_an * diff_an;
      }
    }
  } else if (sizeof(scalar_t) == 8) {
    const int vec_size_d = 2;
    int numVec = D / vec_size_d;
    remainder = D % vec_size_d;
    for (int i = tid; i < numVec; i += blockDim.x) {
      const double2* a_vec = reinterpret_cast<const double2*>(a_ptr);
      const double2* p_vec = reinterpret_cast<const double2*>(p_ptr);
      const double2* n_vec = reinterpret_cast<const double2*>(n_ptr);
      double2 a_val = __ldg(&a_vec[i]);
      double2 p_val = __ldg(&p_vec[i]);
      double2 n_val = __ldg(&n_vec[i]);
#pragma unroll
      for (int j = 0; j < vec_size_d; j++) {
        double a_elem = ((double*)&a_val)[j];
        double p_elem = ((double*)&p_val)[j];
        double n_elem = ((double*)&n_val)[j];
        double diff_ap = a_elem - p_elem;
        double diff_an = a_elem - n_elem;
        partial_ap += diff_ap * diff_ap;
        partial_an += diff_an * diff_an;
      }
    }
  }
  
  // Process any remaining elements that were not vectorized.
  int start = 0;
  if (sizeof(scalar_t) == 4)
    start = (D - remainder);
  else if (sizeof(scalar_t) == 8)
    start = (D - remainder);
  else
    start = 0;  // Fallback if not float/double.
  
  for (int idx = start + tid; idx < D; idx += blockDim.x) {
    scalar_t a_val = a_ptr[idx];
    scalar_t p_val = p_ptr[idx];
    scalar_t n_val = n_ptr[idx];
    scalar_t diff_ap = a_val - p_val;
    scalar_t diff_an = a_val - n_val;
    partial_ap += diff_ap * diff_ap;
    partial_an += diff_an * diff_an;
  }

  // Warp-level reduction via shuffle.
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_ap += __shfl_down_sync(mask, partial_ap, offset);
    partial_an += __shfl_down_sync(mask, partial_an, offset);
  }
  // Each warp stores its reduced sum into shared memory.
  if ((tid & 31) == 0) {
    warp_sdata_ap[warp_id] = partial_ap;
    warp_sdata_an[warp_id] = partial_an;
  }
  __syncthreads();

  // Final reduction across warps.
  if (tid < (BLOCK_SIZE / 32)) {
    partial_ap = warp_sdata_ap[tid];
    partial_an = warp_sdata_an[tid];
    for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset /= 2) {
      partial_ap += __shfl_down_sync(mask, partial_ap, offset);
      partial_an += __shfl_down_sync(mask, partial_an, offset);
    }
    if (tid == 0) {
      scalar_t dist_ap = sqrt(partial_ap);
      scalar_t dist_an = sqrt(partial_an);
      scalar_t loss = dist_ap - dist_an + margin;
      loss_array[sample] = (loss > 0) ? loss : 0;
    }
  }
}

template <typename scalar_t>
torch::Tensor triplet_margin_loss_opt_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

  const int N = anchor.size(0);
  const int D = anchor.size(1);

  constexpr int BLOCK_SIZE = 256;
  dim3 blocks(N);
  dim3 threads(BLOCK_SIZE);

  auto loss_tensor = torch::empty({N}, anchor.options());

  triplet_loss_opt_kernel<scalar_t, BLOCK_SIZE><<<blocks, threads>>>(
      anchor.data_ptr<scalar_t>(),
      positive.data_ptr<scalar_t>(),
      negative.data_ptr<scalar_t>(),
      D,
      static_cast<scalar_t>(margin),
      loss_tensor.data_ptr<scalar_t>());
  return loss_tensor.mean();
}

torch::Tensor triplet_margin_loss_opt_cuda_wrapper(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
  return AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_opt_cuda", ([&] {
    return triplet_margin_loss_opt_cuda<scalar_t>(anchor, positive, negative, margin);
  }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &triplet_margin_loss_opt_cuda_wrapper, "Optimized Triplet margin loss (CUDA) with vectorized loads");
}