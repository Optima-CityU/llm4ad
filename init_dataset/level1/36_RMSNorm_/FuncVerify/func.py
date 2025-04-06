import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        eps (float): A small value added to the denominator to avoid division by zero.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    # Calculate the RMS along the feature dimension
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)

    # Normalize the input by dividing by the RMS
    return x / rms


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]