import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        weight (torch.Tensor): The filter weights for the transposed convolution.
        bias (torch.Tensor): The bias applied to the output.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.weight = conv_transpose2d.weight
        self.bias = conv_transpose2d.bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2


def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]