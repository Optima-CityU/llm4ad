// Cosine Similarity Loss CUDA Kernel

#include <torch/extension.h>
#include <cmath>

__global__ void cosine_similarity_kernel(
    const float* predictions,
    const float* targets,
    float* output,
    int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float dot_product = 0.0f;
        float pred_norm = 0.0f;
        float target_norm = 0.0f;

        // Calculate cosine similarity for each pair of vectors
        for (int d = 0; d < D; d++) {
            float p = predictions[idx * D + d];
            float t = targets[idx * D + d];
            dot_product += p * t;
            pred_norm += p * p;
            target_norm += t * t;
        }

        pred_norm = sqrtf(pred_norm);
        target_norm = sqrtf(target_norm);
        float cosine_sim = dot_product / (pred_norm * target_norm);

        // Compute loss (1 - cosine similarity) for the current sample
        output[idx] = 1.0f - cosine_sim;
    }
}

at::Tensor cosine_similarity_loss_cuda(
    at::Tensor predictions,
    at::Tensor targets
) {
    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = at::empty({N}, predictions.options());

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    cosine_similarity_kernel<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}

at::Tensor cosine_similarity_loss_forward(
    at::Tensor predictions,
    at::Tensor targets
) {
    auto cosine_sim_losses = cosine_similarity_loss_cuda(predictions, targets);
    return cosine_sim_losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss (CUDA)");
}