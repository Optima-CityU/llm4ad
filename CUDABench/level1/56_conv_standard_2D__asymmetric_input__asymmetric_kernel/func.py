import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int) -> torch.Tensor:
    """
    Performs the 2D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): The convolution kernel weights.
        bias (torch.Tensor): The learnable bias tensor.
        stride (tuple): Tuple of two integers representing the stride in the height and width dimensions.
        padding (tuple): Tuple of two integers representing the padding in the height and width dimensions.
        dilation (tuple): Tuple of two integers representing the dilation in the height and width dimensions.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight = conv.weight
        self.bias = conv.bias

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization