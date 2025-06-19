# **Physics Knowledge Discovery** for **Stress & Strain**

#### **Problem**

The Stress & Strain problem, proposed in LLM-SR: Scientific Equation Discovery via Programming with Large Language Models, is a physics-based task that focuses on discovering relationships by minimizing mean square error using environmental
parameters.

```{image} ./stress.png
:width: 80%
:align: center
```

+ **Given:** Environment parameters, a set of constant parameters.

+ **Objective:** Minimize the mean square error.

+ **Constraints:**
    - None

#### Algorithm Design Task

+ **The task** is to design the function to fit the dataset.
    + **Inputs:** Strain, temperature, numeric constants or parameters to be optimized.
    + **Outputs:** Predicted value.

#### Evaluation

+ **Dataset:** Dataset from **LLM-SR: Scientific Equation Discovery via Programming with Large Language Models**.

+ **Fitness:** Mean Square Error

#### Template:

```python
template_program = '''
import numpy as np
def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for stress in Aluminium rod
    Args:
        strain: A numpy array representing observations of strain.
        temp: A numpy array representing observations of temperature.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing stress as the result of applying the mathematical function to the inputs.
    """
    return params[0] * strain  +  params[1] * temp
'''

task_description = "Find the mathematical function skeleton that represents stress, given data on strain and temperature in an Aluminium rod for both elastic and plastic regions."



```

