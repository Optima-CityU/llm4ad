import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, depthwise_weight: torch.Tensor, depthwise_bias: torch.Tensor, pointwise_weight: torch.Tensor, pointwise_bias: torch.Tensor, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Performs a depthwise-separable 2D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        depthwise_weight (torch.Tensor): Weight tensor for the depthwise convolution.
        depthwise_bias (torch.Tensor): Bias tensor for the depthwise convolution.
        pointwise_weight (torch.Tensor): Weight tensor for the pointwise convolution.
        pointwise_bias (torch.Tensor): Bias tensor for the pointwise convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    x = F.conv2d(x, depthwise_weight, depthwise_bias, stride=stride, padding=padding, dilation=dilation, groups=x.size(1))
    x = F.conv2d(x, pointwise_weight, pointwise_bias)
    return x


class Model(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.depthwise_weight = depthwise.weight
        self.depthwise_bias = depthwise.bias
        pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.pointwise_weight = pointwise.weight
        self.pointwise_bias = pointwise.bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.depthwise_weight, self.depthwise_bias, self.pointwise_weight, self.pointwise_bias, self.stride, self.padding, self.dilation)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]