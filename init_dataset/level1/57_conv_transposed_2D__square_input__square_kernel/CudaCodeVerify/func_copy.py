import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose2d_weight: nn.Parameter,
    conv_transpose2d_bias: nn.Parameter,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 2D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv_transpose2d_weight (nn.Parameter): Weight tensor for the transposed convolution.
        conv_transpose2d_bias (nn.Parameter): Bias tensor for the transposed convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(
        x,
        conv_transpose2d_weight,
        bias=conv_transpose2d_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

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

        # Extract parameters from nn.ConvTranspose2d
        conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.conv_transpose2d_weight = nn.Parameter(conv_transpose2d.weight.data.clone())
        self.conv_transpose2d_bias = (
            nn.Parameter(conv_transpose2d.bias.data.clone()) if bias else None
        )

        # Store the configuration parameters
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_transpose2d_weight,
            self.conv_transpose2d_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
width = 128
height = 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization