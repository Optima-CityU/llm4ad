import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose3d_weight: torch.Tensor,
    conv_transpose3d_bias: torch.Tensor,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        conv_transpose3d_weight (torch.Tensor): Weight tensor for the 3D transposed convolution.
        conv_transpose3d_bias (torch.Tensor): Bias tensor for the 3D transposed convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_depth, kernel_size_height, kernel_size_width).
        stride (tuple): Stride of the convolution in the form (stride_depth, stride_height, stride_width).
        padding (tuple): Padding applied to the input in the form (padding_depth, padding_height, padding_width).
        output_padding (tuple): Additional size added to one side of the output shape.
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
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()

        # Initialize convolution weights and biases
        conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias
        )
        self.conv_transpose3d_weight = conv_transpose3d.weight
        if bias:
            self.conv_transpose3d_bias = nn.Parameter(conv_transpose3d.bias)
        else:
            self.conv_transpose3d_bias = None

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_transpose3d_weight,
            self.conv_transpose3d_bias,
            self.conv_transpose3d_weight.shape[2:],
            (2, 2, 2),  # Use stride from initialization
            (1, 2, 3),  # Use padding from initialization
            (1, 1, 1),  # Use output padding from initialization
            4,  # Use groups from initialization
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]