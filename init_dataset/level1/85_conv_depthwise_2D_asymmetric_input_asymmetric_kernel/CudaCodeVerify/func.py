import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv2d_weight: torch.Tensor,
    conv2d_bias: torch.Tensor,
    kernel_size_h: int,
    kernel_size_w: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int
) -> torch.Tensor:
    """
    Performs a depthwise 2D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        conv2d_weight (torch.Tensor): Weight tensor for the depthwise convolution.
        conv2d_bias (torch.Tensor): Bias tensor for the depthwise convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int): Stride of the convolution in height dimension.
        stride_w (int): Stride of the convolution in width dimension.
        padding_h (int): Padding applied to the input in height dimension.
        padding_w (int): Padding applied to the input in width dimension.
        dilation_h (int): Spacing between kernel elements in height dimension.
        dilation_w (int): Spacing between kernel elements in width dimension.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(x, conv2d_weight, conv2d_bias, stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=groups)


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()

        # Create a Conv2d module to extract weights and biases
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.kernel_size[0],  # kernel_size_h
            self.conv2d.kernel_size[1],  # kernel_size_w
            self.conv2d.stride[0],       # stride_h
            self.conv2d.stride[1],       # stride_w
            self.conv2d.padding[0],      # padding_h
            self.conv2d.padding[1],      # padding_w
            self.conv2d.dilation[0],     # dilation_h
            self.conv2d.dilation[1],     # dilation_w
            self.conv2d.groups           # groups
        )


# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]