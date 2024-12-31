from __future__ import annotations

import logging
import pickle
import random
from typing import Any, List

from llm4ad.tools.profiler import ProfilerBase

try:
    import numba
    import torch
    import tensorboardX
except ImportError as e:
    logging.info('Please install "torch", "tensorboard", and "tensorboardX" as this script uses TensorboardProfiler.')
    logging.info('Please install "numba", as this script uses numba acceleration.')
    raise e

from llm4ad.base import LLM

from llm4ad.base.evaluate import Evaluation
from llm4ad.method.eoh import EoH, EoHConfig
from llm4ad.method.eoh.profiler import EoHTensorboardProfiler
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
        fake_thought = '{This is a fake thought for the code}\n'
        rand_func = random.choice(self._functions)
        return fake_thought + rand_func


class OBPEvaluator(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self):
        super().__init__(
            use_numba_accelerate=True,
            timeout_seconds=20
        )
        with open('data/weibull_train.pkl', 'rb') as f:
            self._bin_packing_or_train = pickle.load(f)['weibull_5k_train']

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
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


# In EoH, your template heuristic can have no function body,
# as EoH can initialize the population by sampling heuristics from LLM.
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
    pass
'''

task_description = '''
You can input task descriptions to EoH, to let LLM better understanding their task and goal.
'''

# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':
    sampler = FakeSampler()
    evaluator = OBPEvaluator()
    # profiler = EoHTensorboardProfiler(log_dir='logs/eoh_run1')
    profiler = ProfilerBase(log_dir='logs/eoh_run1')
    config = EoHConfig(pop_size=2)
    eoh = EoH(
        task_description=task_description,
        template_program=template_program,
        sampler=sampler,
        profiler=profiler,
        evaluator=evaluator,
        max_sample_nums=10,
        max_generations=10,
        num_samplers=4,
        num_evaluators=4,
        valid_only=True,
        config=config
    )
    eoh.run()
