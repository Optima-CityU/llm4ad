|-
  #include <torch/extension.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <stdexcept>
  
  // This kernel uses shared memory to load a spatial tile of the input for each depth slice.
  // Threads in the block then reuse the tile to sum over the pooling window.
  __global__ void avg_pool3d_shared_kernel(
      const float* __restrict__ input,
      float* __restrict__ output,
      int batch,
      int channels,
      int in_d,
      int in_h,
      int in_w,
      int out_d,
      int out_h,
      int out_w,
      int kernel_size,
      int stride,
      int padding,
      float inv_pool_volume
  ) {
      // Compute output spatial coordinates.
      int w_out = blockIdx.x * blockDim.x + threadIdx.x;
      int h_out = blockIdx.y * blockDim.y + threadIdx.y;
      if (w_out >= out_w || h_out >= out_h)
          return;
  
      // Decode n, c, and d_out from blockIdx.z.
      int idx = blockIdx.z;
      int d_out = idx % out_d;
      idx /= out_d;
      int c = idx % channels;
      int n = idx / channels;
  
      // Compute corresponding starting indices in input.
      int d_start = d_out * stride - padding;
      int h_start = h_out * stride - padding;
      int w_start = w_out * stride - padding;
      
      float sum = 0.0f;
      
      // Precompute shared memory tile dimensions.
      // Each block covers blockDim.(x,y) output positions spaced by "stride", so the tile size is:
      // tile_width  = (blockDim.x - 1) * stride + kernel_size
      // tile_height = (blockDim.y - 1) * stride + kernel_size
      int tile_w = blockDim.x * stride + kernel_size - stride;
      int tile_h = blockDim.y * stride + kernel_size - stride;
  
      // Compute the top-left coordinate of the tile in input space.
      int h_tile = blockIdx.y * blockDim.y * stride - padding;
      int w_tile = blockIdx.x * blockDim.x * stride - padding;
  
      // Determine valid range in the depth dimension.
      int kd_min = (d_start < 0) ? -d_start : 0;
      int kd_max = (in_d - d_start < kernel_size) ? (in_d - d_start) : kernel_size;
  
      const int in_hw = in_h * in_w;
      const int in_dhw = in_d * in_hw;
      const int base_offset = n * channels * in_dhw + c * in_dhw;
  
      // Loop over the pooling window in the depth dimension.
      for (int kd = kd_min; kd < kd_max; ++kd) {
          int d_in = d_start + kd;
          // Use dynamic shared memory to load a spatial tile from the current depth slice.
          extern __shared__ float tile[];
          int tile_size = tile_h * tile_w;
          int tid = threadIdx.y * blockDim.x + threadIdx.x;
          int num_threads = blockDim.x * blockDim.y;
          for (int idx_tile = tid; idx_tile < tile_size; idx_tile += num_threads) {
              int ty = idx_tile / tile_w;
              int tx = idx_tile % tile_w;
              int h_in = h_tile + ty;
              int w_in = w_tile + tx;
              if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                  int global_idx = base_offset + d_in * in_hw + h_in * in_w + w_in;
                  tile[idx_tile] = input[global_idx];
              } else {
                  tile[idx_tile] = 0.0f;
              }
          }
          __syncthreads();
  
          // For the current depth slice, each thread sums over its pooling window
          // from the shared memory tile.
          // The local top-left coordinate of the pooling window in the tile is:
          // (threadIdx.y * stride, threadIdx.x * stride)
          int local_h = threadIdx.y * stride;
          int local_w = threadIdx.x * stride;
          float partial = 0.0f;
          for (int i = 0; i < kernel_size; ++i) {
              for (int j = 0; j < kernel_size; ++j) {
                  int r = local_h + i;
                  int c_idx = local_w + j;
                  // The tile already contains zeros for out-of-bound regions.
                  partial += tile[r * tile_w + c_idx];
              }
          }
          sum += partial;
          __syncthreads();  // Ensure all threads are done before reusing shared memory.
      }
  
      // Write the computed pooled value to the output tensor.
      int out_hw = out_h * out_w;
      int out_dhw = out_d * out_hw;
      int out_index = n * channels * out_dhw + c * out_dhw + d_out * out_hw + h_out * out_w + w_out;
      output[out_index] = sum * inv_pool_volume;
  }
  
  torch::Tensor avg_pool3d_forward_shared(torch::Tensor input, int kernel_size, c10::optional<int> stride_opt, int padding) {
      TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
      int stride = stride_opt.has_value() ? stride_opt.value() : kernel_size;
  
      int batch = input.size(0);
      int channels = input.size(1);
      int in_d = input.size(2);
      int in_h = input.size(3);
      int in_w = input.size(4);
  
      int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
      int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
      int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
  
      auto output = torch::empty({batch, channels, out_d, out_h, out_w}, input.options());
  
      // Define 2D block dimensions for spatial (w, h).
      const int blockDimX = 16;
      const int blockDimY = 16;
      dim3 block(blockDimX, blockDimY);
      // The 3rd grid dimension encodes (n, c, d_out).
      dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y, batch * channels * out_d);
  
      const int pool_volume = kernel_size * kernel_size * kernel_size;
      const float inv_pool_volume = 1.0f / pool_volume;
  
      // Compute dynamic shared memory size: tile_h x tile_w floats.
      int tile_w = blockDimX * stride + kernel_size - stride;
      int tile_h = blockDimY * stride + kernel_size - stride;
      size_t shared_mem_size = tile_w * tile_h * sizeof(float);
  
      avg_pool3d_shared_kernel<<<grid, block, shared_mem_size>>>(
          input.data_ptr<float>(),
          output.data_ptr<float>(),
          batch,
          channels,
          in_d, in_h, in_w,
          out_d, out_h, out_w,
          kernel_size,
          stride,
          padding,
          inv_pool_volume
      );
  
      return output;
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &avg_pool3d_forward_shared, "3D Average Pooling forward (CUDA) with shared memory");
  }