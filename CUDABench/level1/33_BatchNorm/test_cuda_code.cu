
  #include <torch/extension.h>
  #include <cuda_runtime.h>
  
  // Optimized batch normalization kernel precomputing inverse standard deviation.
  __global__ void improved_batch_norm_kernel(
      const float* __restrict__ x, 
      const float* __restrict__ weight, 
      const float* __restrict__ bias, 
      const float* __restrict__ running_mean, 
      const float* __restrict__ running_var, 
      float* __restrict__ output, 
      int N, int C, int H, int W, float momentum, float eps) 
  {
      // Shared memory layout:
      //  shared_weight: [0, C)
      //  shared_bias:   [C, 2C)
      //  shared_mean:   [2C, 3C)
      //  shared_inv_std: [3C, 4C)
      extern __shared__ float shared_data[];
      float* shared_weight  = shared_data;              // size = C
      float* shared_bias    = shared_data + C;          // size = C
      float* shared_mean    = shared_data + 2 * C;        // size = C
      float* shared_inv_std = shared_data + 3 * C;        // size = C
  
      int tid = threadIdx.x;
      // Load parameters into shared memory and precompute inverse standard deviation.
      for (int c = tid; c < C; c += blockDim.x) {
          shared_weight[c]  = weight[c];
          shared_bias[c]    = bias[c];
          shared_mean[c]    = running_mean[c];
          shared_inv_std[c] = rsqrtf(running_var[c] + eps);
      }
      __syncthreads();
  
      int total = N * C * H * W;
      int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  
      if (idx < total) {
          // Load 4 elements at once
          float4 x_vec = reinterpret_cast<const float4*>(x)[idx / 4];
          float4 out_vec;
  
          int CHW = C * H * W;
          int HW  = H * W;
  
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              int pos = idx + i;
              if (pos < total) {
                  // Compute current channel index from flat index.
                  int c = (pos % CHW) / HW;
                  float x_val = ((float*)&x_vec)[i];
                  float normalized = (x_val - shared_mean[c]) * shared_inv_std[c];
                  ((float*)&out_vec)[i] = shared_weight[c] * normalized + shared_bias[c];
              }
          }
  
          // Store using vectorized write if within bounds.
          if (idx + 3 < total) {
              reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
          } else {
              #pragma unroll
              for (int i = 0; i < 4; ++i) {
                  if (idx + i < total)
                      output[idx + i] = ((float*)&out_vec)[i];
              }
          }
      }
  }
  
  torch::Tensor batch_norm_forward(
      torch::Tensor x, 
      torch::Tensor weight, 
      torch::Tensor bias, 
      torch::Tensor running_mean, 
      torch::Tensor running_var, 
      float momentum, 
      float eps) 
  {
      int N = x.size(0);
      int C = x.size(1);
      int H = x.size(2);
      int W = x.size(3);
  
      auto output = torch::zeros_like(x);
  
      int threads_per_block = 256;
      int total_elements = N * C * H * W;
      int num_blocks = (total_elements + threads_per_block * 4 - 1) / (threads_per_block * 4);
  
      // Allocate shared memory space for 4 arrays, each with C floats.
      size_t shared_mem_size = 4 * C * sizeof(float);
  
      improved_batch_norm_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
          x.data_ptr<float>(), 
          weight.data_ptr<float>(), 
          bias.data_ptr<float>(), 
          running_mean.data_ptr<float>(), 
          running_var.data_ptr<float>(), 
          output.data_ptr<float>(), 
          N, C, H, W, momentum, eps
      );
  
      return output;
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &batch_norm_forward, "Optimized Batch Normalization CUDA kernel with precomputed inverse std");
  }