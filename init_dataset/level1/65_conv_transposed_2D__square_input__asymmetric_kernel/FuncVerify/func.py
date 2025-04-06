import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding,
    output_padding,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w).
        bias (torch.Tensor or None): Bias tensor of shape (out_channels,) or None.
        stride (int): Stride of the convolution.
        padding (int or tuple): Padding applied to the input.
        output_padding (int or tuple): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        # Extract parameters for functional call
        self.weight = nn.Parameter(conv.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]