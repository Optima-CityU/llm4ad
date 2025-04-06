import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose3d_weight: torch.Tensor,
    conv_transpose3d_bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 3D convolution using functional calls.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        conv_transpose3d_weight (torch.Tensor): Weight tensor for the transposed 3D convolution.
        conv_transpose3d_bias (torch.Tensor or None): Bias tensor for the transposed 3D convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
    """
    return F.conv_transpose3d(
        x,
        conv_transpose3d_weight,
        bias=conv_transpose3d_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.conv_transpose3d_weight = nn.Parameter(conv_transpose3d.weight.data.clone())
        if bias:
            self.conv_transpose3d_bias = nn.Parameter(conv_transpose3d.bias.data.clone())
        else:
            self.conv_transpose3d_bias = None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_transpose3d_weight,
            self.conv_transpose3d_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]