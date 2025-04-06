import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, bn_weight: nn.Parameter, bn_bias: nn.Parameter, bn_running_mean: nn.Parameter, bn_running_var: nn.Parameter, eps: float, training: bool) -> torch.Tensor:
    """
    Applies Batch Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        bn_weight (nn.Parameter): BatchNorm scale parameter
        bn_bias (nn.Parameter): BatchNorm shift parameter
        bn_running_mean (nn.Parameter): Running mean for BatchNorm
        bn_running_var (nn.Parameter): Running variance for BatchNorm
        eps (float): A small value to avoid division by zero
        training (bool): Whether the model is in training mode or not

    Returns:
        torch.Tensor: Output tensor with Batch Normalization applied, same shape as input
    """
    return F.batch_norm(x, bn_running_mean, bn_running_var, weight=bn_weight, bias=bn_bias, training=training, eps=eps)


class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """

    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        bn = nn.BatchNorm2d(num_features=num_features)
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_running_mean = bn.running_mean
        self.bn_running_var = bn.running_var
        self.eps = bn.eps

    def forward(self, x: torch.Tensor, fn=module_fn, training: bool = True):
        # Ensure all tensors are on the same device
        device = x.device
        return fn(
            x.to(device),
            self.bn_weight.to(device),
            self.bn_bias.to(device),
            self.bn_running_mean.to(device),
            self.bn_running_var.to(device),
            self.eps,
            training
        )


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]