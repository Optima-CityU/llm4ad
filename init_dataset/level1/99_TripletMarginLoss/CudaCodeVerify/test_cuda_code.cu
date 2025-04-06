// triplet_margin_loss_kernel.cu
#include <torch/extension.h>

__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, int N, float margin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float pos_dist = 0.0f;
        float neg_dist = 0.0f;

        // Compute squared distances
        for (int i = 0; i < 128; ++i) { // assuming 128-dimensional embeddings
            pos_dist += (anchor[idx * 128 + i] - positive[idx * 128 + i]) * (anchor[idx * 128 + i] - positive[idx * 128 + i]);
            neg_dist += (anchor[idx * 128 + i] - negative[idx * 128 + i]) * (anchor[idx * 128 + i] - negative[idx * 128 + i]);
        }

        // Calculate the Triplet Margin Loss
        float dist_diff = pos_dist - neg_dist + margin;
        loss[idx] = fmaxf(dist_diff, 0.0f); // max(0, pos_dist - neg_dist + margin)
    }
}

at::Tensor triplet_margin_loss_forward_cuda(at::Tensor anchor, at::Tensor positive, at::Tensor negative, float margin) {
    int N = anchor.size(0); // Number of samples
    auto loss = at::zeros({N}, anchor.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    triplet_margin_loss_kernel<<<blocks, threads>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        loss.data_ptr<float>(),
        N,
        margin
    );

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_forward_cuda, "Triplet Margin Loss (CUDA)");
}