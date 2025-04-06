import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_transpose3d_weight: nn.Parameter, conv_transpose3d_bias: nn.Parameter, kernel_size: tuple, stride: tuple, padding: tuple, output_padding: tuple, groups: int, bias: bool) -> torch.Tensor:
    """
    Performs a transposed 3D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).
        conv_transpose3d_weight (nn.Parameter): Weight for the transposed 3D convolution.
        conv_transpose3d_bias (nn.Parameter): Bias for the transposed 3D convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height).
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        output_padding (tuple): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If `True`, adds a learnable bias to the output.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
    """
    return F.conv_transpose3d(x, conv_transpose3d_weight, conv_transpose3d_bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups)


class Model(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        
        conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
        self.conv_transpose3d_weight = nn.Parameter(conv_transpose3d.weight.data.clone())
        self.conv_transpose3d_bias = nn.Parameter(conv_transpose3d.bias.data.clone()) if bias else None

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.conv_transpose3d_weight, self.conv_transpose3d_bias, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups, self.bias)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization