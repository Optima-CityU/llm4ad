from typing import Any, Literal
from llm4ad.base import Evaluation
from .template import template_program, task_description


class KernelBenchEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=600
        )

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        r"""Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_str: The function in string. You can _ignore this argument when implementation. (See below).
            callable_func: The callable heuristic function. You can call it using `callable_func(args, kwargs)`.
        Return:
            Returns the fitness value.

        Assume that: self.use_numba_accelerate = True, self.use_protected_div = True,
        and self.random_seed = 2024, the argument 'function_str' will be something like below:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f(a, b):
            np.random.seed(2024)
            a = a + np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator,
        and '_protected_dev' will be added by this function.
        """
        raise NotImplementedError('Must provide a evaluator for a function.')