#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// Clamp helper ---------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_unit(scalar_t v) {
    return v < scalar_t(0) ? scalar_t(0) : (v > scalar_t(1) ? scalar_t(1) : v);
}

// half‑precision specialization
template <>
__device__ __forceinline__ at::Half clamp_unit(at::Half v) {
    float f = __half2float(v);
    f = f < 0.f ? 0.f : (f > 1.f ? 1.f : f);
    return __float2half(f);
}

// Kernel ---------------------------------------------------------------------

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ x,
                                   scalar_t* __restrict__ y,
                                   int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // PyTorch formula: clamp((x / 6) + 0.5, 0, 1)
        const scalar_t inv6 = scalar_t(1.0) / scalar_t(6.0);
        scalar_t v = x[idx] * inv6 + scalar_t(0.5);
        y[idx] = clamp_unit(v);
    }
}

// Interface ------------------------------------------------------------------

torch::Tensor hardsigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto output = at::empty_like(input);

    const int64_t numel = input.numel();
    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                        "hardsigmoid_forward_cuda", ([&] {
        hardsigmoid_kernel<scalar_t><<<blocks, threads, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// PyBind ---------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hardsigmoid_forward, "HardSigmoid forward (CUDA)");
}