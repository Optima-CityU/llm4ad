import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the GELU activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the GELU activation.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []