import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs product reduction over the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to reduce over.

    Returns:
        torch.Tensor: Output tensor with product reduction applied.
    """
    return torch.prod(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that performs product reduction over a dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.dim)


batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1


def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [reduction_dim]