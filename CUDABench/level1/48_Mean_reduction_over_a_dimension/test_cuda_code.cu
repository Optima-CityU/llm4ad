|-
  #include <torch/extension.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <vector>
  #include <type_traits>
  
  template <typename scalar_t>
  __global__ void ldg_unroll16_512t_opt_kernel(const scalar_t * __restrict__ input, 
                                               scalar_t * __restrict__ output, 
                                               int outer, int red, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
  
    for (; idx < outer * inner; idx += stride) {
      int i = idx / inner;
      int k = idx % inner;
      // compute pointer to the beginning of the reduction slice for current output element
      const scalar_t *input_ptr = input + i * red * inner + k;
      scalar_t sum = 0;
      // precompute reciprocal so that we do one division per thread
      scalar_t inv_red = static_cast<scalar_t>(1) / static_cast<scalar_t>(red);
  
      // Optimized branch when the inner dimension is 1 (contiguous reduction)
      if (inner == 1) {
          // Use vectorized loads if possible; rely on the fact that underlying scalar_t is either float (4 bytes)
          // or double (8 bytes). For float, if red is a multiple of 4 then use float4; for double, if red is a multiple
          // of 2 then use double2.
          if (sizeof(scalar_t) == 4 && (red & 3) == 0) {
              // vectorized loop for float: each float4 loads 4 floats.
              int vec_len = red / 4;
              const float4* vec_ptr = reinterpret_cast<const float4*>(input_ptr);
              #pragma unroll
              for (int q = 0; q < vec_len; q++) {
                float4 v = __ldg(&vec_ptr[q]);
                sum += v.x + v.y + v.z + v.w;
              }
          } else if (sizeof(scalar_t) == 8 && (red & 1) == 0) {
              // vectorized loop for double: use double2 if possible.
              int vec_len = red / 2;
              const double2* vec_ptr = reinterpret_cast<const double2*>(input_ptr);
              #pragma unroll
              for (int q = 0; q < vec_len; q++) {
                double2 v = __ldg(&vec_ptr[q]);
                sum += v.x + v.y;
              }
          } else {
              // Fall back to scalar loads if vectorization is not applicable.
              #pragma unroll
              for (int j = 0; j < red; j++) {
                  sum += __ldg(&input_ptr[j]);
              }
          }
      } else {
          // Non-contiguous inner dimension route: use unrolling factor 16.
          int j = 0;
          int unroll_limit = (red / 16) * 16;
          #pragma unroll
          for (; j < unroll_limit; j += 16) {
              sum += __ldg(&input_ptr[j * inner]) +
                     __ldg(&input_ptr[(j + 1) * inner]) +
                     __ldg(&input_ptr[(j + 2) * inner]) +
                     __ldg(&input_ptr[(j + 3) * inner]) +
                     __ldg(&input_ptr[(j + 4) * inner]) +
                     __ldg(&input_ptr[(j + 5) * inner]) +
                     __ldg(&input_ptr[(j + 6) * inner]) +
                     __ldg(&input_ptr[(j + 7) * inner]) +
                     __ldg(&input_ptr[(j + 8) * inner]) +
                     __ldg(&input_ptr[(j + 9) * inner]) +
                     __ldg(&input_ptr[(j + 10) * inner]) +
                     __ldg(&input_ptr[(j + 11) * inner]) +
                     __ldg(&input_ptr[(j + 12) * inner]) +
                     __ldg(&input_ptr[(j + 13) * inner]) +
                     __ldg(&input_ptr[(j + 14) * inner]) +
                     __ldg(&input_ptr[(j + 15) * inner]);
          }
          for (; j < red; j++) {
              sum += __ldg(&input_ptr[j * inner]);
          }
      }
  
      output[idx] = sum * inv_red;
    }
  }
  
  torch::Tensor forward(torch::Tensor input, int dim) {
      input = input.contiguous();
      auto sizes = input.sizes();
      int ndim = sizes.size();
  
      int outer = 1;
      for (int i = 0; i < dim; i++) {
          outer *= sizes[i];
      }
      int red = sizes[dim];
      int inner = 1;
      for (int i = dim + 1; i < ndim; i++) {
          inner *= sizes[i];
      }
  
      std::vector<int64_t> out_sizes;
      for (int i = 0; i < ndim; i++) {
          if (i != dim) {
              out_sizes.push_back(sizes[i]);
          }
      }
      auto output = torch::empty(out_sizes, input.options());
  
      int total = outer * inner;
      int threads = 512;
      int blocks = (total + threads - 1) / threads;
  
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ldg_unroll16_512t_opt_kernel", ([&] {
          ldg_unroll16_512t_opt_kernel<scalar_t><<<blocks, threads>>>(
              input.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              outer, red, inner);
      }));
  
      return output;
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &forward, "Mean reduction with LDG, unroll 16, 512 threads optimized (CUDA)");
  }