import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Performs a 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor for convolution.
        bias (torch.Tensor): Bias tensor for convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight = conv.weight
        self.bias = conv.bias

        # Save initialization arguments for functional version
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization