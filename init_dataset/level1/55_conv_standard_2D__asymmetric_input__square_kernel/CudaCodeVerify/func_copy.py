import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv2d_weight: torch.Tensor, conv2d_bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Performs a 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv2d_weight (torch.Tensor): Weight tensor for the 2D convolution.
        conv2d_bias (torch.Tensor): Bias tensor for the 2D convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(x, conv2d_weight, bias=conv2d_bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()

        conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv2d_weight = conv2d.weight
        self.conv2d_bias = conv2d.bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.conv2d_weight, self.conv2d_bias, self.stride, self.padding, self.dilation, self.groups)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128  # Asymmetric input


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization