# name: str: BGEvaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.science_discovery.bactgrow.template import template_program, task_description
from llm4ad.task.science_discovery.bactgrow import train

__all__ = ['BGEvaluation']

MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS


def evaluate(data: dict, equation: callable) -> float | None:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    b, s, temp, pH = inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3]

    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(b, s, temp, pH, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')

    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss


class BGEvaluation(Evaluation):

    def __init__(self, timeout_seconds=20, **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        # read csv
        # df = pd.read_csv(os.path.join(os.path.dirname(__file__), './_data/train.csv'))
        df = pd.DataFrame(train.data)
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        self._datasets = {'inputs': X, 'outputs': y}

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self._datasets, callable_func)

if __name__ == '__main__':
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

    eval = BGEvaluation()
    res = eval.evaluate_program('', equation)
    print(res)
