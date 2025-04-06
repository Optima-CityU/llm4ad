import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Performs a 1D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        conv_weight (torch.Tensor): The weights of the convolutional layer.
        conv_bias (torch.Tensor): The bias of the convolutional layer.
        stride (int): The stride of the convolution.
        padding (int): The padding applied to the input.
        dilation (int): The spacing between kernel elements.
        groups (int): The number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
    """
    return F.conv1d(x, conv_weight, conv_bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_weight = conv1d.weight
        self.conv_bias = conv1d.bias if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.stride, self.padding, self.dilation, self.groups)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 512


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization