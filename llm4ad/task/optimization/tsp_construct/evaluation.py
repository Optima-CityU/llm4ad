# name: str: TSPEvaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tsp_construct.get_instance import GetData
from llm4ad.task.optimization.tsp_construct.template import template_program, task_description

__all__ = ['TSPEvaluation']

def tour_cost(instance, solution, problem_size):
    cost = 0
    for j in range(problem_size - 1):
        cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
    cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
    return cost

def generate_neighborhood_matrix(instance):
    instance = np.array(instance)
    n = len(instance)
    neighborhood_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        distances = np.linalg.norm(instance[i] - instance, axis=1)
        sorted_indices = np.argsort(distances)  # sort indices based on distances
        neighborhood_matrix[i] = sorted_indices

    return neighborhood_matrix

def evaluate(instance_data,n_ins,prob_size, eva: callable) -> float:

    n_max = n_ins
    dis = np.ones(n_ins)
    n_ins = 0

    for instance, distance_matrix in instance_data:

        # get neighborhood matrix
        neighbor_matrix = generate_neighborhood_matrix(instance)

        destination_node = 0

        current_node = 0

        route = np.zeros(prob_size)
        # print(">>> Step 0 : select node "+str(instance[0][0])+", "+str(instance[0][1]))
        for i in range(1, prob_size - 1):

            near_nodes = neighbor_matrix[current_node][1:]

            mask = ~np.isin(near_nodes, route[:i])

            unvisited_near_nodes = near_nodes[mask]

            next_node = eva(current_node, destination_node, unvisited_near_nodes, distance_matrix)

            if next_node in route:
                # print("wrong algorithm select duplicate node, retrying ...")
                return None

            current_node = next_node

            route[i] = current_node


        mask = ~np.isin(np.arange(prob_size), route[:prob_size - 1])

        last_node = np.arange(prob_size)[mask]

        current_node = last_node[0]

        route[prob_size - 1] = current_node

        # print(">>> Step "+str(self.problem_size-1)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))

        LLM_dis = tour_cost(instance, route, prob_size)
        dis[n_ins] = LLM_dis

        n_ins += 1
        if n_ins == n_max:
            break
        # self.route_plot(instance,route,self.oracle[n_ins])

    ave_dis = np.average(dis)
    # print("average dis: ",ave_dis)
    return -ave_dis


class TSPEvaluation(Evaluation):
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

        self.n_instance = 16
        self.problem_size = 50
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets = getData.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self._datasets,self.n_instance,self.problem_size, callable_func)
    

if __name__ == '__main__':

    def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int: 
        """
        Design a novel algorithm to select the next node in each step.

        Args:
        current_node: ID of the current node.
        destination_node: ID of the destination node.
        unvisited_nodes: Array of IDs of unvisited nodes.
        distance_matrix: Distance matrix of nodes.

        Return:
        ID of the next node to visit.
        """
        distances_to_destination = distance_matrix[current_node][unvisited_nodes]

        # Find the index of the unvisited node with the smallest distance to the destination
        next_node_index = np.argmin(distances_to_destination)

        # Get the ID of the next node to visit
        next_node = unvisited_nodes[next_node_index]

        return next_node
    
    tsp = TSPEvaluation()
    tsp.evaluate_program('_',select_next_node)