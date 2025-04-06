import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Performs the 3D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).
        conv_weight (torch.Tensor): The convolution weights tensor.
        conv_bias (torch.Tensor): The convolution bias tensor.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
    """
    return F.conv3d(x, conv_weight, conv_bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()

        conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.conv_weight, self.conv_bias, self.stride, self.padding, self.dilation, self.groups)


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization