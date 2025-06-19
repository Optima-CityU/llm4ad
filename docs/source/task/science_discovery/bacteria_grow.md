# **Biology Knowledge Discovery** for **Bacteria Growth**

#### **Problem** 
The Bacteria Growth problem, introduced in LLM-SR: Scientific Equation Discovery via Programming with Large Language Models, is a biology-focused task aiming to discover growth patterns by minimizing mean square error based on environmental parameters.

```{image} ./biology.png
:width: 80%
:align: center
```

+ **Given:** Bacteria environment parameters, a set of constant parameters.

+ **Objective:** Minimize the mean square error.

+ **Constraints:** 
    - None


#### Algorithm Design Task

+ **The task** is to design the function to fit the dataset.
  + **Inputs:** Population density of the bacterial species, substrate concentration, temperature, PH level, numeric constants or parameters to be optimized.
  + **Outputs:** Predicted value.

#### Evaluation

+ **Dataset:** Dataset from **LLM-SR: Scientific Equation Discovery via Programming with Large Language Models**. 

+ **Fitness:** Mean Square Error


#### Template: 

```python
template_program = '''
import numpy as np
def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for bacterial growth rate
    Args:
        b: A numpy array representing observations of population density of the bacterial species.
        s: A numpy array representing observations of substrate concentration.
        temp: A numpy array representing observations of temperature.
        pH: A numpy array representing observations of pH level.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.
    """
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]

'''

task_description = "Find the mathematical function skeleton that represents E. Coli bacterial growth rate, given data on population density, substrate concentration, temperature, and pH level."


```

