import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D tensor-matrix multiplication.

    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).

    Returns:
        torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


N = 16
M = 1024
K = 2048
L = 768


def get_inputs():
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed