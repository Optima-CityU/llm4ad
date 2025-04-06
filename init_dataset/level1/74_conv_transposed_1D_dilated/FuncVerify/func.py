import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Performs the transposed 1D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length)
        weight (torch.Tensor): The convolutional kernel weights
        bias (torch.Tensor): The learnable bias
        stride (int): The stride of the convolution
        padding (int): The padding applied to the input
        dilation (int): The dilation of the convolution kernel

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out)
    """
    return F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()

        # Initialize parameters using the ConvTranspose1d module
        conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Extract parameters from the convolutional layer
        self.weight = nn.Parameter(conv_transpose.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(conv_transpose.bias.data.clone())
        else:
            self.bias = None

        # Store convolutional parameters to use them in forward pass
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]