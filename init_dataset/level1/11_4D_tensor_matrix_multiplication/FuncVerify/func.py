import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A, B):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    return torch.einsum("bijl,lk->bijk", A, B)


class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


b = 16
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.randn(b, i, j, l, device='cuda')
    B = torch.randn(l, k, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed