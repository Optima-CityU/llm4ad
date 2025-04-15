import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, gn_weight: torch.Tensor, gn_bias: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Applies Group Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        gn_weight (torch.Tensor): Group Normalization weight tensor
        gn_bias (torch.Tensor): Group Normalization bias tensor
        num_groups (int): Number of groups to divide the channels into

    Returns:
        torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
    """
    return F.group_norm(x, num_groups, weight=gn_weight, bias=gn_bias)


class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """

    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.gn_weight = self.gn.weight
        self.gn_bias = self.gn.bias

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.gn_weight, self.gn_bias, self.gn.num_groups)


batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, num_groups]