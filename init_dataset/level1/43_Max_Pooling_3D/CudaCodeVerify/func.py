import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    return_indices: bool,
    ceil_mode: bool,
) -> torch.Tensor:
    """
    Applies Max Pooling 3D to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).
        kernel_size (int): Size of the kernel for the max pooling operation.
        stride (int): Stride of the pooling operation.
        padding (int): Padding applied to the input tensor.
        dilation (int): Spacing between kernel elements.
        return_indices (bool): Whether to return indices of the maximum values.
        ceil_mode (bool): When True, the output size is ceil(input_size / stride) instead of floor.

    Returns:
        torch.Tensor: Output tensor with Max Pooling 3D applied.
    """
    return F.max_pool3d(
        x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode,
        )


batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]