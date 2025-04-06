import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose3d_weight: nn.Parameter,
    conv_transpose3d_bias: nn.Parameter,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Performs a 3D transposed convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose3d_weight (nn.Parameter): Weight tensor for the 3D transposed convolution
        conv_transpose3d_bias (nn.Parameter): Bias tensor for the 3D transposed convolution
        stride (int): Stride for the transposed convolution
        padding (int): Padding for the transposed convolution
        dilation (int): Dilation for the transposed convolution

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
    """
    return F.conv_transpose3d(
        x, conv_transpose3d_weight, conv_transpose3d_bias, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False
    ):
        super(Model, self).__init__()

        conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.conv_transpose3d_weight = conv_transpose3d.weight
        self.conv_transpose3d_bias = conv_transpose3d.bias

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_transpose3d_weight,
            self.conv_transpose3d_bias,
            self.stride,
            self.padding,
            self.dilation,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]