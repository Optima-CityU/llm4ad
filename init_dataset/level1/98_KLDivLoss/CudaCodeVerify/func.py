import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler Divergence for comparing two distributions.

    Args:
        predictions (torch.Tensor): The predicted probability distribution.
        targets (torch.Tensor): The target probability distribution.

    Returns:
        torch.Tensor: The Kullback-Leibler Divergence between predictions and targets.
    """
    return F.kl_div(torch.log(predictions), targets, reduction='batchmean')


class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]


def get_init_inputs():
    return []