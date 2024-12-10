# name: str: OscillatorEvaluation1
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.science_discovery.oscillator1.template import template_program, task_description
from llm4ad.task.science_discovery.oscillator1 import train

__all__ = ['OscillatorEvaluation1']

MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS


def evaluate(data: dict, equation: callable) -> float | None:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    x, v = inputs[:, 0], inputs[:, 1]

    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(x, v, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0] * MAX_NPARAMS, method='BFGS')

    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss


class OscillatorEvaluation1(Evaluation):

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
    def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        """ Mathematical function for acceleration in a damped nonlinear oscillator
        Args:
            x: A numpy array representing observations of current position.
            v: A numpy array representing observations of velocity.
            params: Array of numeric constants or parameters to be optimized

        Return:
            A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
        """
        dv = params[0] * x + params[1] * v + + params[3]
        return dv

    eval = OscillatorEvaluation1()
    res = eval.evaluate_program('', equation)
    print(res)
