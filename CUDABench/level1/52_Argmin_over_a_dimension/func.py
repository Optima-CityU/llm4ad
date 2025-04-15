import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Finds the index of the minimum value along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to find the minimum value.

    Returns:
        torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
    """
    return torch.argmin(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.dim)


batch_size = 16
dim1 = 256
dim2 = 256
dim = 1


def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [dim]