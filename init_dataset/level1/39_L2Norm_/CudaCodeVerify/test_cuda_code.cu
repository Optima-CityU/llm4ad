#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void forward_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               const int N,  // Number of rows (all dimensions except dim=1)
                               const int D)  // Size of dim=1
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int offset = row * D;
        float sum = 0.0f;
        // Compute the squared L2 norm for the row.
        for (int j = 0; j < D; j++) {
            float val = input[offset + j];
            sum += val * val;
        }
        float norm = sqrtf(sum);
        // Normalize each element in the row.
        for (int j = 0; j < D; j++) {
            output[offset + j] = input[offset + j] / norm;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    // Ensure the input has at least two dimensions.
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    // Normalize along dimension 1.
    int D = input.size(1);
    // Flatten all dimensions except the normalization dimension.
    int N = input.numel() / D;

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        N, D);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 Normalization forward (CUDA)");
}