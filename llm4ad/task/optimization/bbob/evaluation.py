

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Callable, Any, List, Tuple
import copy

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bbob.template import template_program, task_description
from llm4ad.task.optimization.bbob.GNBG.GNBG_instances import GNBG

__all__ = ['BBOBEvaluation']


class BBOBEvaluation(Evaluation):

    def __init__(self,
                 timeout_seconds: int = 60,
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


    def evaluate(self, eva: Callable) -> float:
        """
        Evaluate the constructive heuristic for the 1D Bin Packing Problem.

        Args:
            instance_data: List of tuples containing the item weights and bin capacity.
            n_ins: Number of instances to evaluate.
            eva: The constructive heuristic function to evaluate.
            n_bins: The number of available bins at the beginning.

        Returns:
            The average number of bins used across all instances.
        """
        total_bins = 0

        for instance in self._datasets:
            item_weights, bin_capacity = instance
            num_bins, _ = self.pack_items(item_weights, bin_capacity, eva, self.n_bins)
            total_bins += num_bins

        average_bins = total_bins / self.n_instance
        return -average_bins  # Negative because we want to minimize the number of bins

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)


if __name__ == '__main__':

    def determine_next_assignment(remaining_items: List[int], remaining_capacities: List[int]) -> Tuple[int, int | None]:
        """
        Determine the next item and bin to pack based on a greedy heuristic.

        Args:
            remaining_items: A list of remaining item weights.
            remaining_capacities: A list of remaining capacities of feasible bins.

        Returns:
            A tuple containing:
            - The selected item to pack.
            - The selected bin to pack the item into (or None if no feasible bin is found).
        """
        # Simple greedy heuristic: choose the largest item that fits into the bin with the smallest remaining capacity
        for item in sorted(remaining_items, reverse=True):  # Try largest items first
            for bin_id, capacity in enumerate(remaining_capacities):
                if item <= capacity:
                    return item, bin_id  # Return the selected item and bin
        return remaining_items[0], None  # If no feasible bin is found, return the first item and no bin
