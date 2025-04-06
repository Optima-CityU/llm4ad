import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(anchor, positive, negative, margin):
    """
    Computes the Triplet Margin Loss.

    Args:
        anchor (torch.Tensor): The anchor tensor.
        positive (torch.Tensor): The positive sample tensor.
        negative (torch.Tensor): The negative sample tensor.
        margin (float): The margin between the positive and negative samples.

    Returns:
        torch.Tensor: The computed Triplet Margin Loss.
    """
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)


class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    """

    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, fn=module_fn):
        return fn(anchor, positive, negative, self.margin)


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin