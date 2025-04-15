import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, momentum: float, eps: float) -> torch.Tensor:
    """
    Applies Batch Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        weight (torch.Tensor): Scale factor for the normalization
        bias (torch.Tensor): Bias added after normalization
        running_mean (torch.Tensor): Running mean for BatchNorm
        running_var (torch.Tensor): Running variance for BatchNorm
        momentum (float): Momentum for updating the running statistics
        eps (float): Small constant added to the variance for numerical stability

    Returns:
        torch.Tensor: Output tensor with Batch Normalization applied, same shape as input
    """
    return F.batch_norm(x, running_mean, running_var, weight, bias, training=True, momentum=momentum, eps=eps)


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

        # Initialize BatchNorm parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x: torch.Tensor, fn=module_fn):
        # Ensure all tensors are on the same device (e.g., CUDA or CPU)
        device = x.device
        weight = self.weight.to(device)
        bias = self.bias.to(device)
        running_mean = self.running_mean.to(device)
        running_var = self.running_var.to(device)

        return fn(x, weight, bias, running_mean, running_var, self.momentum, self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]