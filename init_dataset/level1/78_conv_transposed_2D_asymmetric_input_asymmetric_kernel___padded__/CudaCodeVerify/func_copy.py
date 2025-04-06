import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_transpose2d_weight: nn.Parameter, conv_transpose2d_bias: nn.Parameter, stride: tuple, padding: tuple) -> torch.Tensor:
    """
    Performs the 2D transposed convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv_transpose2d_weight (nn.Parameter): Weight of the transposed convolution.
        conv_transpose2d_bias (nn.Parameter): Bias of the transposed convolution.
        stride (tuple): Stride of the convolution (height, width).
        padding (tuple): Padding applied to the input (height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(x, conv_transpose2d_weight, conv_transpose2d_bias, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(Model, self).__init__()
        conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_transpose2d_weight = nn.Parameter(conv_transpose2d.weight.data.clone())
        self.conv_transpose2d_bias = nn.Parameter(conv_transpose2d.bias.data.clone()) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, self.conv_transpose2d_weight, self.conv_transpose2d_bias, self.stride, self.padding)


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]