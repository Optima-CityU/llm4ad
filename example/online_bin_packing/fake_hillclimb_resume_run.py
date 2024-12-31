from __future__ import annotations

import logging
import pickle
import random
from typing import Any, List

from llm4ad.method.hillclimb.resume import resume_hillclimb

try:
    import numba
    import torch
    import tensorboardX
except ImportError as e:
    logging.info('Please install "torch", "tensorboard", and "tensorboardX" as this script uses TensorboardProfiler.')
    logging.info('Please install "numba", as this script uses numba acceleration.')
    raise e

from llm4ad.base import LLM
from llm4ad.tools.profiler import TensorboardProfiler
from llm4ad.base.evaluate import Evaluation
from llm4ad.method.hillclimb import HillClimb
from _obp_evaluate import evaluate


class FakeSampler(LLM):
    """We select random functions from rand_function.pkl
    This sampler can help you debug your method even if you don't have an LLM API / deployed local LLM.
    """

    def __init__(self):
        super().__init__()
        with open('./data/rand_function.pkl', 'rb') as f:
            self._functions = pickle.load(f)

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        return super().draw_samples(prompts)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        return random.choice(self._functions)


class OBPEvaluator(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self):
        super().__init__(
            use_numba_accelerate=True,
            timeout_seconds=20
        )
        with open('data/weibull_train.pkl', 'rb') as f:
            self._bin_packing_or_train = pickle.load(f)['weibull_5k_train']

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        """Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_str: The function in string. You can _ignore this argument when implementation.
            callable_func: The callable Python function of your sampled heuristic function code.
            You can call the program using 'program_callable(args..., kwargs...)'
        Return:
            Returns the fitness value. Return None if you think the result is invalid.
        """
        # we call the _obp_evaluate.evaluate function to evaluate the callable code
        return evaluate(self._bin_packing_or_train, callable_func)


template_program = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    penalty = np.arange(len(bins), 0, -1)
    scores = bins / (bins - item) - penalty
    max_capacity_bins = np.where(bins == bins.max())[0]
    for idx in max_capacity_bins:
        scores[idx] = -np.inf
    return scores
'''

# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':
    sampler = FakeSampler()
    evaluator = OBPEvaluator()
    profiler = TensorboardProfiler(log_dir='logs/run1', initial_num_samples=0)
    hillclimb = HillClimb(
        template_program=template_program,
        sampler=sampler,
        profiler=profiler,
        evaluator=evaluator,
        max_sample_nums=100,
        num_samplers=4,
        num_evaluators=4
    )

    # ================== Simply invoke the resume function ==================
    resume_hillclimb(hillclimb)
    # =======================================================================

    hillclimb.run()
