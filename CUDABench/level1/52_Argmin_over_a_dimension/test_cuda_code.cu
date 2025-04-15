#include <torch/extension.h>
#include <vector>

// Kernel: computes argmin along the specified dimension (assuming x is contiguous).
// We flatten all dimensions except 'dim' into two parts: [left_size, dim_size, right_size].
// For each of the left_size * right_size positions, we iterate over dim_size to find the min index.
__global__ void argmin_kernel(const float* __restrict__ x,
                              long* __restrict__ out,
                              const int64_t left_size,
                              const int64_t right_size,
                              const int64_t dim_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = left_size * right_size;
    if (idx < total) {
        // Map idx to (row, col).
        int64_t row = idx / right_size;
        int64_t col = idx % right_size;

        // Base offset for this "row" along 'dim'.
        int64_t base = row * dim_size * right_size + col;

        float min_val = x[base];
        long min_idx = 0;
        // Search along dim_size.
        for (int64_t i = 1; i < dim_size; i++) {
            float val = x[base + i * right_size];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        out[idx] = min_idx;
    }
}

// Forward function: computes argmin along 'dim', returning an int64 output tensor
// with 'dim' removed from the shape.
torch::Tensor forward(torch::Tensor x, int64_t dim) {
    // Ensure float32 for simplicity.
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32.");

    // Compute sizes.
    int64_t dim_size = x.size(dim);
    int64_t left_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        left_size *= x.size(i);
    }
    int64_t right_size = 1;
    for (int64_t i = dim + 1; i < x.dim(); i++) {
        right_size *= x.size(i);
    }

    // Build output shape: same as x.sizes() but remove dimension 'dim'.
    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim() - 1);
    for (int64_t i = 0; i < x.dim(); i++) {
        if (i != dim) {
            out_shape.push_back(x.size(i));
        }
    }

    // Create output tensor (int64).
    auto out = torch::empty(out_shape, x.options().dtype(torch::kInt64));

    // Launch kernel.
    int64_t total = left_size * right_size;
    const int64_t threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;

    argmin_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<long>(),
        left_size,
        right_size,
        dim_size
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Argmin forward (CUDA)");
}