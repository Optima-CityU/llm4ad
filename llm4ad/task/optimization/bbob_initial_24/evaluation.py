

from __future__ import annotations
from typing import Callable, Any, List, Tuple

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bbob_initial.template import template_program, task_description
from llm4ad.task.optimization.bbob_initial.core import *

__all__ = ['BBOBEvaluationINI24']


class BBOBEvaluationINI24(Evaluation):

    def __init__(self,
                 timeout_seconds: int = 1800,
                 **kwargs):
        """
        Args:
            n_bins: The number of available bins at the beginning.
        """
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        # Get the current script's directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to the folder where you want to read/write files
        self.folder_path = os.path.join(current_dir)


    def evaluate(self, eva: Callable, program_str: str) -> float:
        num_process = 10
        total_runs = 10

        fitness = main(num_process=num_process, total_runs=total_runs, test_problems=[24],
                       random_seed=[2025 + i for i in range(total_runs)], func=program_str)

        return -fitness  # Negative because we want to minimize the number of bins

    def evaluate_program(self, program_str: str, callable_func: Callable, **kwargs) -> Any | None:
        return self.evaluate(callable_func, program_str)



