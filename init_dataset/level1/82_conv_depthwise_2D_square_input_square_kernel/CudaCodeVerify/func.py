import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv_weight (torch.Tensor): Weight tensor for the convolution.
        conv_bias (torch.Tensor): Bias tensor for the convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
    """
    return F.conv2d(x, conv_weight, conv_bias, stride=stride, padding=padding, groups=x.shape[1])


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    """

    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        
        conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.conv_weight, self.conv_bias, self.stride, self.padding)


# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]