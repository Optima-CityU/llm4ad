import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    """
    Applies Max Pooling 2D to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        kernel_size (int): Size of the pooling window.
        stride (int): Stride of the pooling window.
        padding (int): Padding to be applied before pooling.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
    """
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, *get_init_inputs())


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]