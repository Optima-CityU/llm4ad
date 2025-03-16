class FuncConvertPrompts:
    SYS_PROMPT = \
    f"You are an expert python and PyTorch engineer.\n\nYour job is correctness and holding to the given task specification."
    @staticmethod
    def get_conversion_prompt(code_to_convert: str) -> str:
        prompt_content = """
You will be given python code, which contains four parts:
- [Imports] These are always at the top of the file and contains mostly torch imports.
- [Model Definition] A `Model(nn.Module)` class with an init and a forward method.
- [Configurations] Following the `Model` class, there are some configurations and two functions `get_inputs()` and `get_init_inputs()`. The configurations are used in these functions.
- [Members] The `Model` class may have members that are also `nn.Module`, they are defined either in `torch.nn` or before the `Model` class.

Your job is to write a "functional" version of that code, which suggests the workflow of `Model(*get_init_inputs())(*get_inputs())`.
The task for each part of the code is as follows:

For [Imports], you will likely need the following libraries:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
You may import other libraries, but you probably not any other torch modules.
For example, if you do `from torch import _VF`, you can then use `_VF.lstm` and `_VF.gru` functions.

For [Model Definition], you should:
- Define `nn.Parameter` in `Model.__init__()`, since they are needed for the functional calls.
    - If a parameter is already defined in the `Model` class, you can directly use it.
    - Otherwise, you may extract it from the `nn.Module`s. E.g. `self.conv.weights = nn.Parameter(self.conv1.weight.data.clone)`, where `self.conv1` is an `nn.Module`. If you use `nn.ParameterDict` to extract parameters, remember that names cannot contain dots.
    - Notice that these parameters' specifications are given by the `get_init_inputs()` function call because Model is initialized with `Model(*get_init_inputs())`.
- Modify the `forward()` method to use functional calls instead of `nn.Modules`.
    - This can be achieved by defining a `module_fn()` that takes in all the args that the forward pass was called with as well as any neural network parameters that the Model holds. It should faithfully reproduce the exact forward pass as the original Model class.
    - Then, you can augment the `Model.forward()` method signature, where you add a `fn` argument that defaults to `module_fn`.
- You need to read the original code carefully. E.g., the code may not implement a residual connection even though the name or comment suggests it.

For [Configurations], you should simply copy the code from the original file. They are for test purposes.
For [Members], you should:
- If a member is an `nn.Module` for which there exists a functional call, you can replace its forward pass with the functional call. E.g., if it is a `nn.Conv2d`, you can replace `self.conv(x)` with `F.conv2d(x, ...)` with proper arguments.
- Otherwise, you need to decompose it until the its forward pass can be replaced with a series of functional calls. You can define functions for this, and these functions will be used in the `module_fn()` function.

Your returned functional version of the code should be a valid python file, and it will be checked against the original code. Their outputs should be identical.
Return only python code, no other text.
The following are several examples to illustrate the task.
"""
        example_1 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return torch.diag(A) @ B

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A, B):
    """
    Performs a matrix multiplication of a diagonal matrix with another matrix.

    Args:
        A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
        B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

    Returns:
        torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
    """
    return torch.diag(A) @ B


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 4096
N = 4096


def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
```
'''
        example_2 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # Add the same noise as in functional implementation
        self.conv.bias = nn.Parameter(self.conv.bias + torch.ones_like(self.conv.bias) * 0.02)
        self.group_norm.bias = nn.Parameter(self.group_norm.bias + torch.ones_like(self.group_norm.bias) * 0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies 3D convolution, group normalization, and computes mean.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        conv_weight (torch.Tensor): 3D convolution weight tensor
        conv_bias (torch.Tensor): 3D convolution bias tensor
        group_norm_weight (torch.Tensor): Group norm weight tensor
        group_norm_bias (torch.Tensor): Group norm bias tensor
        num_groups (int): Number of groups for group normalization

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1)
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.group_norm(x, num_groups, weight=group_norm_weight, bias=group_norm_bias)
    x = x.mean(dim=[1, 2, 3, 4])
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )  # make sure its nonzero
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )  # make sure its nonzero

    def forward(self, x, num_groups, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            num_groups,
        )


batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
num_groups = 8


def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W), num_groups]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
```
        '''
        example_3 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, weights: nn.ParameterList, biases: nn.ParameterList
) -> torch.Tensor:
    """
    Implements a deep narrow multi-layer perceptron with ReLU activation.

    Args:
        x (torch.Tensor): The input tensor, shape (batch_size, input_size)
        weights (nn.ParameterList): A list of weight tensors for each linear layer
        biases (nn.ParameterList): A list of bias tensors for each linear layer

    Returns:
        torch.Tensor: The output tensor, shape (batch_size, output_size)
    """
    for weight, bias in zip(weights[:-1], biases[:-1]):
        x = F.linear(x, weight, bias)
        x = F.relu(x)
    x = F.linear(x, weights[-1], biases[-1])
    return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            linear = nn.Linear(current_input_size, hidden_size)
            self.weights.append(nn.Parameter(linear.weight.data.clone()))
            self.biases.append(nn.Parameter(linear.bias.data.clone()))
            current_input_size = hidden_size

        linear = nn.Linear(current_input_size, output_size)
        self.weights.append(nn.Parameter(linear.weight.data.clone()))
        self.biases.append(nn.Parameter(linear.bias.data.clone()))

    def forward(self, x, fn=module_fn):
        return fn(x, self.weights, self.biases)


# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
]  # Example of deep and narrow layers
output_size = 10


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
```
        '''
        example_4 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv1_weight: nn.Parameter,
    conv1_bias: nn.Parameter,
    conv2_weight: nn.Parameter,
    conv2_bias: nn.Parameter,
    fc1_weight: nn.Parameter,
    fc1_bias: nn.Parameter,
    fc2_weight: nn.Parameter,
    fc2_bias: nn.Parameter,
    fc3_weight: nn.Parameter,
    fc3_bias: nn.Parameter,
) -> torch.Tensor:
    """
    Implements a LeNet-5 architecture with ReLU activation.

    Args:
        x (torch.Tensor): The input tensor, shape (batch_size, 1, 32, 32)
        conv1_weight (nn.Parameter): Parameters for first conv layer
        conv1_bias (nn.Parameter): Parameters for first conv layer
        conv2_weight (nn.Parameter): Parameters for second conv layer
        conv2_bias (nn.Parameter): Parameters for second conv layer
        fc1_weight (nn.Parameter): Parameters for first FC layer
        fc1_bias (nn.Parameter): Parameters for first FC layer
        fc2_weight (nn.Parameter): Parameters for second FC layer
        fc3_weight (nn.Parameter): Parameters for third FC layer
        fc3_bias (nn.Parameter): Parameters for third FC layer

    Returns:
        torch.Tensor: The output tensor, shape (batch_size, num_classes)
    """
    # First convolutional layer with ReLU activation and max pooling
    x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Second convolutional layer with ReLU activation and max pooling
    x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Flatten the output for the fully connected layers
    x = x.view(-1, 16 * 5 * 5)

    # First fully connected layer with ReLU activation
    x = F.linear(x, fc1_weight, fc1_bias)
    x = F.relu(x)

    # Second fully connected layer with ReLU activation
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.relu(x)

    # Final fully connected layer
    x = F.linear(x, fc3_weight, fc3_bias)

    return x


class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()

        # Extract parameters from convolutional layers
        conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv1_weight = nn.Parameter(conv1.weight.data.clone())
        self.conv1_bias = nn.Parameter(conv1.bias.data.clone())

        conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv2_weight = nn.Parameter(conv2.weight.data.clone())
        self.conv2_bias = nn.Parameter(conv2.bias.data.clone())

        # Extract parameters from fully connected layers
        fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc1_weight = nn.Parameter(fc1.weight.data.clone())
        self.fc1_bias = nn.Parameter(fc1.bias.data.clone())

        fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc2_weight = nn.Parameter(fc2.weight.data.clone())
        self.fc2_bias = nn.Parameter(fc2.bias.data.clone())

        fc3 = nn.Linear(in_features=84, out_features=num_classes)
        self.fc3_weight = nn.Parameter(fc3.weight.data.clone())
        self.fc3_bias = nn.Parameter(fc3.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias,
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.fc3_weight,
            self.fc3_bias,
        )


# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10


def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]


def get_init_inputs():
    return [num_classes]
```
        '''
        prompt_content += f"""

=== Example 1 ===

{example_1}

=== Example 2 ===

{example_2}

=== Example 3 ===

{example_3}

=== Example 4 ===

{example_4}

=================
"""
        prompt_content += f"""

Here is the code you need to convert:

```python
{code_to_convert}
```

Your functional version of the code should be a valid python file, and it will be checked against the original code. Their outputs should be identical.
Return only python code, no other text.
"""
        return prompt_content

    @staticmethod
    def get_error_prompt(error_message: str) -> str:
        content_new = f"""
The above functional code does not work as expected. Error message:

{error_message}

Please provide the correct functional code.
Your returned functional version of the code should be a valid python file, and it will be checked against the original code. Their outputs should be identical.
Return only python code, no other text.
"""
        return content_new
