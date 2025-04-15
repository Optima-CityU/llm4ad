import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies Max reduction over the specified dimension to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension to reduce over.

    Returns:
        torch.Tensor: Output tensor after Max reduction over the specified dimension.
    """
    return torch.max(x, dim=dim)[0]


class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.dim)


batch_size = 16
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [1]  # Example, change to desired dimension