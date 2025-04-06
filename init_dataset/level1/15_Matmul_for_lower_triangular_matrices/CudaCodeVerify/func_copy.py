import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of lower triangular matrices A and B.

    Args:
        A (torch.Tensor): Lower triangular matrix of shape (N, N).
        B (torch.Tensor): Lower triangular matrix of shape (N, N).

    Returns:
        torch.Tensor: The result of matrix multiplication C of shape (N, N).
    """
    return torch.tril(torch.matmul(A, B))


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed