import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Hinge Loss for binary classification tasks.

    Args:
        predictions (torch.Tensor): The predicted values, shape (batch_size, 1).
        targets (torch.Tensor): The target labels, shape (batch_size, 1), with values in {-1, 1}.

    Returns:
        torch.Tensor: The computed hinge loss.
    """
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))


class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 128
input_shape = (1,)
dim = 1


def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]


def get_init_inputs():
    return []