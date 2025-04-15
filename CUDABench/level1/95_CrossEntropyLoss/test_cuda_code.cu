|  
  #include <torch/extension.h>  
  #include <cuda_runtime.h>  
  #include <cmath>  
  
  #define BLOCK_SIZE 256  
  // Assume max iterations per thread does not exceed 8 (suitable for typical C values)  
  #define MAX_ITERS 8  
  
  __global__ void cross_entropy_register_kernel(const float* __restrict__ predictions,  
                                                 const int64_t* __restrict__ targets,  
                                                 float* __restrict__ loss, int N, int C) {  
      int sample_id = blockIdx.x;  
      if (sample_id >= N) return;  
  
      const float* sample_preds = predictions + sample_id * C;  
  
      // load target index once per block  
      __shared__ int64_t s_target;  
      if (threadIdx.x == 0) {  
          s_target = targets[sample_id];  
      }  
      __syncthreads();  
  
      // Each thread loads a few elements from sample_preds and stores them in registers.  
      // This avoids reading the same elements twice.  
      int num_iters = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;  
      // For typical classification problems, num_iters is small (e.g., <= 8)  
      float local_vals[MAX_ITERS];  
      float local_max = -INFINITY;  
      for (int i = 0; i < num_iters; i++) {  
          int idx = threadIdx.x + i * BLOCK_SIZE;  
          float val = (idx < C) ? __ldg(sample_preds + idx) : -INFINITY;  
          local_vals[i] = val;  
          local_max = fmaxf(local_max, val);  
      }  
  
      // Warp-level reduction for maximum  
      for (int offset = 16; offset > 0; offset >>= 1) {  
          local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));  
      }  
  
      // Shared memory for per-warp max reduction  
      __shared__ float s_max[BLOCK_SIZE/32];  
      if ((threadIdx.x & 31) == 0) {  
          s_max[threadIdx.x >> 5] = local_max;  
      }  
      __syncthreads();  
  
      float sample_max = -INFINITY;  
      if (threadIdx.x < BLOCK_SIZE/32) {  
          float val = s_max[threadIdx.x];  
          // intra-warp reduction among warp leaders  
          for (int offset = 16; offset > 0; offset >>= 1) {  
              val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));  
          }  
          if (threadIdx.x == 0) {  
              sample_max = val;  
              s_max[0] = val;  // store for broadcast  
          }  
      }  
      __syncthreads();  
      sample_max = s_max[0];  
  
      // Now compute the sum of exp(val - sample_max) using values stored in registers  
      float local_sum = 0.0f;  
      for (int i = 0; i < num_iters; i++) {  
          int idx = threadIdx.x + i * BLOCK_SIZE;  
          if (idx < C) {  
              local_sum += expf(local_vals[i] - sample_max);  
          }  
      }  
  
      // Warp-level reduction for sum  
      for (int offset = 16; offset > 0; offset >>= 1) {  
          local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);  
      }  
  
      // Shared memory for per-warp sum reduction  
      __shared__ float s_sum[BLOCK_SIZE/32];  
      if ((threadIdx.x & 31) == 0) {  
          s_sum[threadIdx.x >> 5] = local_sum;  
      }  
      __syncthreads();  
  
      float sum_exp = 0.0f;  
      if (threadIdx.x < BLOCK_SIZE/32) {  
          float val = s_sum[threadIdx.x];  
          for (int offset = 16; offset > 0; offset >>= 1) {  
              val += __shfl_down_sync(0xffffffff, val, offset);  
          }  
          if (threadIdx.x == 0) {  
              sum_exp = val;  
              s_sum[0] = val;  
          }  
      }  
      __syncthreads();  
      sum_exp = s_sum[0];  
  
      // Load target prediction value once using the target index  
      float target_val = 0.0f;  
      if (threadIdx.x == 0) {  
          target_val = __ldg(sample_preds + s_target);  
      }  
      __syncthreads();  
  
      if (threadIdx.x == 0) {  
          loss[sample_id] = sample_max + logf(sum_exp) - target_val;  
      }  
  }  
  
  torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {  
      const int N = predictions.size(0);  
      const int C = predictions.size(1);  
      torch::Tensor loss = torch::zeros({N}, predictions.options());  
      cross_entropy_register_kernel<<<N, BLOCK_SIZE>>>(predictions.data_ptr<float>(),  
                                                       targets.data_ptr<int64_t>(),  
                                                       loss.data_ptr<float>(), N, C);  
      return loss.mean();  
  }  
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  
      m.def("forward", &cross_entropy_cuda, "Cross Entropy Loss (CUDA) with Register Optimization");  
  }