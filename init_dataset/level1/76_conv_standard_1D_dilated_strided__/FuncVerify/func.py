import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x,
    conv1d_weight,
    conv1d_bias,
    stride,
    dilation,
):
    """
    Performs the 1D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        conv1d_weight (torch.Tensor): Weight tensor of shape (out_channels, in_channels, kernel_size).
        conv1d_bias (torch.Tensor or None): Bias tensor of shape (out_channels,) or None.
        stride (int): Stride of the convolution.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
    """
    return F.conv1d(x, conv1d_weight, bias=conv1d_bias, stride=stride, dilation=dilation)


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super(Model, self).__init__()
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.conv1d_weight = nn.Parameter(conv.weight.data.clone())
        self.conv1d_bias = nn.Parameter(conv.bias.data.clone()) if conv.bias is not None else None
        self.stride = stride
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv1d_weight, self.conv1d_bias, self.stride, self.dilation)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 256
stride = 3
dilation = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]