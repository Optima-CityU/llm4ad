import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, output_padding: int, groups: int) -> torch.Tensor:
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        weight (torch.Tensor): Weight tensor for the convolution.
        bias (torch.Tensor): Bias tensor for the convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups)


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        
        conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.weight = conv_transpose2d.weight
        self.bias = conv_transpose2d.bias

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups)


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]