import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, normalized_shape: tuple) -> torch.Tensor:
    """
    Applies Layer Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        ln_weight (torch.Tensor): The learned weight parameter of the LayerNorm.
        ln_bias (torch.Tensor): The learned bias parameter of the LayerNorm.
        normalized_shape (tuple): Shape of the input tensor to be normalized.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    return F.layer_norm(x, normalized_shape, weight=ln_weight, bias=ln_bias)


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
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.ln_weight = self.ln.weight
        self.ln_bias = self.ln.bias

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.ln_weight, self.ln_bias, self.ln.normalized_shape)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]