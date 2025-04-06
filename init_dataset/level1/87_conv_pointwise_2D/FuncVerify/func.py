import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a pointwise 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv_weight (torch.Tensor): Weight tensor for the 2D convolution.
        conv_bias (torch.Tensor): Bias tensor for the 2D convolution.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
    """
    return F.conv2d(x, conv_weight, bias=conv_bias, stride=1, padding=0)


class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias if bias else None

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.conv_weight, self.conv_bias)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]