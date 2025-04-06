import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 2D transposed convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor for the transposed convolution.
        bias (torch.Tensor): Bias tensor for the transposed convolution.
        stride (tuple): Stride of the convolution (height, width).
        padding (tuple): Padding applied to the input (height, width).
        dilation (tuple): Spacing between kernel elements (height, width).
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(
        x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
    )


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()

        conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.weight = conv_transpose2d.weight
        self.bias = conv_transpose2d.bias if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]