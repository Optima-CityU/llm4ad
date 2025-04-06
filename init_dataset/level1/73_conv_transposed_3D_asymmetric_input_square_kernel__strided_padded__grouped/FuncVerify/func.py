import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose3d_weight: nn.Parameter,
    conv_transpose3d_bias: nn.Parameter,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs the 3D transposed convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose3d_weight (nn.Parameter): Weight parameters for the 3D transposed convolution
        conv_transpose3d_bias (nn.Parameter): Bias parameters for the 3D transposed convolution
        stride (int): Stride of the convolution
        padding (int): Padding applied to the input
        output_padding (int): Padding applied to the output
        groups (int): Number of blocked connections from input channels to output channels

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
    """
    return F.conv_transpose3d(
        x, conv_transpose3d_weight, bias=conv_transpose3d_bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups
    )


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()

        conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias
        )
        
        self.conv_transpose3d_weight = conv_transpose3d.weight
        self.conv_transpose3d_bias = conv_transpose3d.bias

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn):
        # Adjust the output_padding to ensure it does not exceed stride values
        output_padding = tuple(min(s, p) for s, p in zip(self.stride, self.output_padding))
        return fn(
            x,
            self.conv_transpose3d_weight,
            self.conv_transpose3d_bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
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
padding = 3
groups = 4


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]