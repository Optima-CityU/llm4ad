import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Applies 2D Average Pooling to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        kernel_size (int): Size of the pooling window.
        stride (int): Stride of the pooling operation.
        padding (int): Padding applied to the input tensor.

    Returns:
        torch.Tensor: Output tensor with Average Pooling applied.
    """
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.kernel_size, self.stride, self.padding)


batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size]