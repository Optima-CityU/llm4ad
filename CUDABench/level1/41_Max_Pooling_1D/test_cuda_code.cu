|  
  #include <torch/extension.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cfloat>
  #include <tuple>
  
  // This kernel loads the entire input slice for one (batch,feature) pair to shared memory,
  // then each thread computes one output element by doing the max pooling over the required window.
  __global__ void optimized_max_pool1d_kernel(
      const float* __restrict__ input, 
      float* __restrict__ output, 
      int* __restrict__ indices,
      const int batch_size, 
      const int num_features, 
      const int sequence_length, 
      const int kernel_size, 
      const int stride, 
      const int padding, 
      const int dilation, 
      const bool return_indices
  ) {
      // Each block processes one (batch, feature) slice.
      int b = blockIdx.x; // batch index
      int f = blockIdx.y; // feature index
  
      // Compute output sequence length.
      int output_sequence_length = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  
      // Allocate shared memory (dynamically sized) for one input slice.
      extern __shared__ float tile[]; // size: sequence_length elements
  
      // Each thread loads part of the input slice into shared memory.
      for (int i = threadIdx.x; i < sequence_length; i += blockDim.x) {
          tile[i] = input[(b * num_features + f) * sequence_length + i];
      }
      __syncthreads();
  
      // Each thread computes one output element if within range.
      int o = threadIdx.x;
      if (o < output_sequence_length) {
          int input_start = o * stride - padding;
          int window_end = input_start + kernel_size * dilation;
  
          float max_val = -FLT_MAX;
          int max_idx = -1;
  
          // Loop over the pooling window with step = dilation.
          for (int i = input_start; i < window_end; i += dilation) {
              if (i >= 0 && i < sequence_length) {
                  float val = tile[i];
                  if (val > max_val) {
                      max_val = val;
                      max_idx = i;
                  }
              }
          }
  
          output[(b * num_features + f) * output_sequence_length + o] = max_val;
          if (return_indices) {
              indices[(b * num_features + f) * output_sequence_length + o] = max_idx;
          }
      }
  }
  
  at::Tensor max_pool1d_cuda(
      at::Tensor input, 
      int kernel_size, 
      int stride, 
      int padding, 
      int dilation, 
      bool return_indices
  ) {
      int batch_size = input.size(0);
      int num_features = input.size(1);
      int sequence_length = input.size(2);
  
      int output_sequence_length = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  
      at::Tensor output = at::empty({batch_size, num_features, output_sequence_length}, input.options());
      at::Tensor indices = at::empty({batch_size, num_features, output_sequence_length}, input.options().dtype(at::kInt));
  
      // Choose block size: ensure we have enough threads to cover both the shared load and the output computation.
      int block_threads = output_sequence_length;
      // In case output_sequence_length is small, we use at least 128 threads for cooperative loading.
      if (block_threads < 128) {
          block_threads = 128;
      }
  
      dim3 block(block_threads);
      dim3 grid(batch_size, num_features);
  
      // Allocate shared memory: one float per element of the input slice.
      size_t shared_mem_size = sequence_length * sizeof(float);
  
      optimized_max_pool1d_kernel<<<grid, block, shared_mem_size>>>(
          input.data_ptr<float>(), 
          output.data_ptr<float>(), 
          indices.data_ptr<int>(), 
          batch_size, 
          num_features, 
          sequence_length, 
          kernel_size, 
          stride, 
          padding, 
          dilation, 
          return_indices
      );
  
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
      }
  
      if (return_indices) {
          return at::native::ivalue_to_tensor(std::make_tuple(output, indices));
      }
      return output;
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &max_pool1d_cuda, "Optimized 1D max pooling (CUDA)");
  }