import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Cross Entropy Loss for multi-class classification tasks.

    Args:
        predictions (torch.Tensor): Predicted logits, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth labels, shape (batch_size,)

    Returns:
        torch.Tensor: Cross Entropy Loss value
    """
    return F.cross_entropy(predictions, targets)


class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 4096
num_classes = 10
input_shape = (num_classes,)  # Output for each class
dim = 1


def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]


def get_init_inputs():
    return []