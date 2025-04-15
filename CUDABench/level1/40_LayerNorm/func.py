import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float
) -> torch.Tensor:
    """
    Applies Layer Normalization to the input tensor using functional calls.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        normalized_shape (tuple): The shape over which to normalize.
        weight (torch.Tensor): The learnable weight tensor for LayerNorm.
        bias (torch.Tensor): The learnable bias tensor for LayerNorm.
        eps (float): A value added to the denominator for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = normalized_shape
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.normalized_shape, self.weight, self.bias, self.eps)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]