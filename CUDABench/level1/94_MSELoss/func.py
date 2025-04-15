import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error loss for regression tasks.

    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The true target values.

    Returns:
        torch.Tensor: The computed mean squared error loss.
    """
    return torch.mean((predictions - targets) ** 2)


class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []