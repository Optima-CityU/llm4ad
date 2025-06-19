# **Math Knowledge Discovery** for **Oscillator2**

#### **Problem** 
The Oscillator2 problem, introduced in LLM-SR: Scientific Equation Discovery via Programming with Large Language Models, is a mathematical exploration task focused on identifying oscillator patterns by minimizing mean square error with given environmental parameters.


```{image} ./oscillator2.png
:width: 80%
:align: center
```

+ **Given:** Environment parameters, a set of constant parameters.

+ **Objective:** Minimize the mean square error.

+ **Constraints:** 
    - None


#### Algorithm Design Task

+ **The task** is to design the function to fit the dataset.
  + **Inputs:** Time, current position, velocity, numeric constants or parameters to be optimized.
  + **Outputs:** Predicted value.

#### Evaluation

+ **Dataset:** Dataset from **LLM-SR: Scientific Equation Discovery via Programming with Large Language Models**. 

+ **Fitness:** Mean Square Error


#### Template: 

```python
template_program = '''
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator
    Args:
        t: A numpy array representing time.
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * t + params[1] * x  +  params[2] * v +  + params[3]
    return dv


'''

task_description = "Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on time, position, and velocity."


```

