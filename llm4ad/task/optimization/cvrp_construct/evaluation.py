# name: str: CVRPEvaluation
# Parameters: 
# timeout_seconds: int: 20
# end
from __future__ import annotations

import copy
from typing import Any

import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.cvrp_construct.get_instance import GetData
from llm4ad.task.optimization.cvrp_construct.template import template_program, task_description


class CVRPEvaluation(Evaluation):
    def __init__(self,timeout_seconds=20,  **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.problem_size = 50
        self.n_instance = 16

        # path = os.path.join(os.path.dirname(__file__), './_data/data.pkl')
        # with open(path, 'rb') as f:
        #     data = pickle.load(f)
        # self.instance_data = data

        getData = GetData(self.n_instance, self.problem_size+1)
        self._datasets = getData.generate_instances()
        

    def tour_cost(self, instance, solution):
        cost = 0
        for j in range(len(solution) - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost


    def evaluate(self, heuristic):
        dis = np.ones(self.n_instance)
        n_ins = 0

        for instance, distance_matrix, demands, vehicle_capacity in self._datasets:
            route = []
            current_load = 0
            current_node = 0
            route.append(current_node)

            unvisited_nodes = set(range(1, self.problem_size+1))  # Assuming node 0 is the depot
            all_nodes = np.array(list(unvisited_nodes))
            feasible_unvisited_nodes = all_nodes

            while unvisited_nodes:
                next_node = heuristic(current_node,
                                      0,
                                      feasible_unvisited_nodes,  # copy
                                      vehicle_capacity - current_load,
                                      copy.deepcopy(demands),  # copy
                                      copy.deepcopy(distance_matrix))  # copy
                if next_node == 0:
                    # Update route and load
                    route.append(next_node)
                    current_load = 0
                    current_node = 0
                else:
                    # Update route and load
                    route.append(next_node)
                    current_load += demands[next_node]
                    unvisited_nodes.remove(next_node)
                    current_node = next_node

                feasible_nodes_capacity = np.array([node for node in all_nodes if current_load + demands[node] <= vehicle_capacity])
                # Determine feasible and unvisited nodes
                feasible_unvisited_nodes = np.intersect1d(feasible_nodes_capacity, list(unvisited_nodes))
                

                if len(unvisited_nodes) > 0 and len(feasible_unvisited_nodes) < 1:
                    route.append(0)
                    current_load = 0
                    current_node = 0
                    feasible_unvisited_nodes = np.array(list(unvisited_nodes))

            # check if not all nodes have been visited 
            independent_values = set(route)
            if len(independent_values) != self.problem_size+1:
                return None

            LLM_dis = self.tour_cost(instance, route)
            dis[n_ins] = LLM_dis

            n_ins += 1
            if n_ins == self.n_instance:
                break

        ave_dis = np.average(dis)
        return -ave_dis


    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

if __name__ == '__main__':
    # def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    #     """Design a novel algorithm to select the next node in each step.
    #     Args:
    #         current_node: ID of the current node.
    #         depot: ID of the depot.
    #         unvisited_nodes: Array of IDs of unvisited nodes.
    #         rest_capacity: rest capacity of vehicle
    #         demands: demands of nodes
    #         distance_matrix: Distance matrix of nodes.
    #     Return:
    #         ID of the next node to visit.
    #     """
    #     next_node = unvisited_nodes[0]
    #     return next_node

    def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
        """Design a novel algorithm to select the next node in each step.
        Args:
            current_node: ID of the current node.
            depot: ID of the depot.
            unvisited_nodes: Array of IDs of unvisited nodes.
            rest_capacity: rest capacity of vehicle
            demands: demands of nodes
            distance_matrix: Distance matrix of nodes.
        Return:
            ID of the next node to visit.
        """
        best_score = -1
        next_node = -1

        for node in unvisited_nodes:
            demand = demands[node]
            distance = distance_matrix[current_node][node]

            if demand <= rest_capacity:
                score = demand / distance if distance > 0 else float('inf')  # Avoid division by zero
                if score > best_score:
                    best_score = score
                    next_node = node

        return next_node

    eval = CVRPEvaluation()
    res = eval.evaluate_program('', select_next_node)
    print(res)
