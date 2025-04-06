import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_transpose2d_weight: torch.Tensor, conv_transpose2d_bias: torch.Tensor, kernel_size: tuple, stride: tuple, padding: tuple, output_padding: tuple, dilation: tuple, groups: int) -> torch.Tensor:
    """
    Performs a transposed 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        conv_transpose2d_weight (torch.Tensor): Weight tensor for the transposed convolution.
        conv_transpose2d_bias (torch.Tensor): Bias tensor for the transposed convolution.
        kernel_size (tuple): Tuple representing the kernel size (height, width).
        stride (tuple): Tuple representing the stride of the convolution.
        padding (tuple): Tuple representing the padding applied to the input.
        output_padding (tuple): Tuple representing the additional size added to one side of the output shape.
        dilation (tuple): Tuple representing the spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(x, conv_transpose2d_weight, conv_transpose2d_bias, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        
        # Initialize the convolution transpose parameters
        conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_transpose2d_weight = conv_transpose2d.weight
        self.conv_transpose2d_bias = conv_transpose2d.bias if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_transpose2d_weight,
            self.conv_transpose2d_bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]