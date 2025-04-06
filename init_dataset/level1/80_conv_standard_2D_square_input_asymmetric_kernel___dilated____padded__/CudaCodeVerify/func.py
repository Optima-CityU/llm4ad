import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    stride: int,
    padding: tuple,
    dilation: tuple,
) -> torch.Tensor:
    """
    Performs a 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv_weight (torch.Tensor): The convolutional weights.
        conv_bias (torch.Tensor): The convolutional bias.
        stride (int): Stride of the convolution.
        padding (tuple): Padding applied to the input (top/bottom, left/right).
        dilation (tuple): Spacing between kernel elements (height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(x, conv_weight, conv_bias, stride=stride, padding=padding, dilation=dilation)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.stride,
            self.padding,
            self.dilation,
        )


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256
stride = 1
padding = (1, 2)  # Asymmetric padding
dilation = (2, 1)  # Asymmetric dilation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]