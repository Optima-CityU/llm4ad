# name: str: OBP_2O_Evaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.online_bin_packing.template import template_program, task_description
from llm4ad.task.optimization.online_bin_packing.generate_weibull_instances import generate_weibull_dataset

import time
from typing import Tuple

__all__ = ['OBP_2O_Evaluation']


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray, priority: callable
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate(instances: dict, priority: callable) -> np.ndarray:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []

    start_time = time.time()

    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins, priority)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    running_time = time.time() - start_time
    return np.array([-np.mean(num_bins), -running_time/len(instances)])


class OBP_2O_Evaluation(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self, timeout_seconds=60, data_file='weibull_train.pkl', data_key='weibull_5k_train', **kwargs):
        """
        Args:
            - 'data_file' (str): The data file to load (default is 'weibull_5k_train.pkl').
            - 'data_key' (str): The key of the data to load (default is 'data_key').

        Raises:
            AttributeError: If the data key does not exist.
            FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self._datasets = generate_weibull_dataset(5, 5000, 100)

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self._datasets, callable_func)


if __name__ == '__main__':
    import numpy as np


    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        """Returns priority with which we want to add item to each bin.
        Args:
            item: Size of item to be added to the bin.
            bins: Array of capacities for each bin.
        Return:
            Array of same size as bins with priority score of each bin.
        """
        return -bins


    bpp = OBP_2O_Evaluation()
    bpp.evaluate_program('_', priority)