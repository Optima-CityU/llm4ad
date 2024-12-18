import sys
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.base import Evaluation
from template import template_program, task_description
from gls import guided_local_search_with_time
from typing import Tuple, Any

class TSPInstance:
    def __init__(self, positions: npt.NDArray[np.float_]) -> None:
        self.positions = positions
        self.n = positions.shape[0]
        self.distmat = distance_matrix(positions, positions) + np.eye(self.n)*1e-5


perturbation_moves = 5
iter_limit = 1000


def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()


def solve_with_time(inst: TSPInstance, eva) -> Tuple[float, float]:
    try:
        result, running_time = guided_local_search_with_time(inst.distmat, inst.distmat.copy(), eva, perturbation_moves, iter_limit)
        cost = calculate_cost(inst, result)
    except Exception as e:
        # cost, running_time = 1E10, 1E10
        cost, running_time = float("inf"), float("inf")
    # print(result)
    return cost, running_time


def evaluate(instance_data,n_ins,prob_size, eva: callable) -> np.ndarray:
    objs = np.zeros((n_ins, 2))

    for i in range(n_ins):
        obj = solve_with_time(instance_data[i], eva)
        objs[i] = np.array(obj)

    obj = np.mean(objs, axis=0)
    return -obj


class TSP_GLS_2O_Evaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self, **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )

        self.n_instance = 100
        self.problem_size = 100
        self._datasets = [TSPInstance(d) for d in np.random.random((self.n_instance, self.problem_size, 2))]

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self._datasets,self.n_instance,self.problem_size, callable_func)


def main():
    llm = HttpsApi(host="xxx",  # your host endpoint, e.g., api.openai.com, api.deepseek.com
                   key="xxx",  # your key, e.g., sk-xxxxxxxxxx
                   model="gpt-3.5-turbo",  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=60)

    task = TSP_GLS_2O_Evaluation()

    num_objs = 2

    method = MEoH(llm=llm,
                  profiler=MEoHProfiler(log_dir='logs', log_style='complex'),
                  evaluation=task,
                  max_sample_nums=20000,
                  max_generations=5,
                  pop_size=20,
                  num_samplers=1,
                  num_evaluators=1,
                  num_objs=num_objs)

    method.run()


if __name__ == '__main__':
    main()
