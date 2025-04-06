import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a reverse cumulative sum operation along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the reverse cumulative sum.

    Returns:
        torch.Tensor: The result after reverse cumulative sum along the specified dimension.
    """
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)


class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim)


batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs():
    return [torch.randn(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]