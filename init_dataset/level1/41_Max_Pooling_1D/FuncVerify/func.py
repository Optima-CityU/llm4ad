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
) -> torch.Tensor:
    return F.max_pool1d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
        )


batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]