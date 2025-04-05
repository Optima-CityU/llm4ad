import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies ELU activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        alpha (float): The alpha parameter for the ELU function.

    Returns:
        torch.Tensor: Output tensor with ELU applied, same shape as input.
    """
    return F.elu(x, alpha=alpha)


class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.alpha)


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization