import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, inorm_weight: torch.Tensor, inorm_bias: torch.Tensor) -> torch.Tensor:
    """
    Applies Instance Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
        inorm_weight (torch.Tensor): InstanceNorm weight tensor.
        inorm_bias (torch.Tensor): InstanceNorm bias tensor.

    Returns:
        torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
    """
    return F.instance_norm(x, weight=inorm_weight, bias=inorm_bias)


class Model(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """

    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()

        # Define the parameters for InstanceNorm
        self.inorm_weight = nn.Parameter(torch.ones(num_features))
        self.inorm_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, fn=module_fn):
        return fn(x, self.inorm_weight, self.inorm_bias)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]