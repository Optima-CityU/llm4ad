import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...).

    Returns:
        torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
    """
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True)


class Model(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []