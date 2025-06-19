# **Math Knowledge Discovery** for **Oscillator1**

#### **Problem** 
The Oscillator1 problem, presented in LLM-SR: Scientific Equation Discovery via Programming with Large Language Models, is a mathematical task aimed at uncovering oscillator behaviors by minimizing mean square error using environmental parameters.

```{image} ./oscillator1.png
:width: 80%
:align: center
```
+ **Given:** Environment parameters, a set of constant parameters.

+ **Objective:** Minimize the mean square error.

+ **Constraints:** 
    - None


#### Algorithm Design Task

+ **The task** is to design the function to fit the dataset.
  + **Inputs:** Current position, velocity, numeric constants or parameters to be optimized.
  + **Outputs:** Predicted value.

#### Evaluation

+ **Dataset:** Dataset from **LLM-SR: Scientific Equation Discovery via Programming with Large Language Models**. 

+ **Fitness:** Mean Square Error


#### Template: 

```python
template_program = '''
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator
    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * x  +  params[1] * v +  + params[3]
    return dv

'''

task_description = "Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity."


```

