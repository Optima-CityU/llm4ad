import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Applies 1D Average Pooling to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).
        kernel_size (int): Size of the pooling window.
        stride (int): Stride of the pooling operation.
        padding (int): Padding applied to the input tensor.

    Returns:
        torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
    """
    return F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.kernel_size, self.stride, self.padding)


batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]