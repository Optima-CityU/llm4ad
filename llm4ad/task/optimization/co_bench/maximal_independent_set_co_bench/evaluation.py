
from __future__ import annotations

import os
import pathlib
import pickle
from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.co_bench.maximal_independent_set_co_bench.template import template_program, task_description

__all__ = ['MISEvaluationCB']


class MISEvaluationCB(Evaluation):

    def __init__(self,
                 timeout_seconds=50,
                 **kwargs):

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
            timeout_seconds=timeout_seconds
        )

        path = os.path.dirname(os.path.abspath(__file__))
        ins_files_path = os.listdir(os.path.join(path, 'ins/er_test'))  # er_large_test, er_test
        self._datasets = [os.path.join(path, 'ins/er_test', e) for e in ins_files_path if e.endswith('.gpickle')]

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, eva: callable) -> float | None:
        ins_cases = []
        for case_id, ins in enumerate(self._datasets):
            ins_cases.append(self.load_data(ins))

        fitness_list = []
        try:
            for i in ins_cases:
                for j in i:
                    result = eva(j['graph'])
                    fitness = self.eval_func(name=j['name'], graph=j['graph'], mis_nodes=result['mis_nodes'], mis_size=len(result['mis_nodes']))
                    fitness_list.append(fitness)

            return np.mean(fitness_list)

        except ValueError as e:
            print(e)
            return None

    def load_data(self, file_path):
        """
        Load test data for MIS problem from a single file or directory.
        Args:
            file_path (str or pathlib.Path): Path to a .gpickle file or directory containing .gpickle files
        Returns:
            list: A list of dictionaries, each containing a test case with graph data
        """
        file_path = pathlib.Path(file_path)
        test_cases = []

        # Function to process a single graph file
        def process_graph_file(graph_path):
            try:
                # Manual loading using pickle if nx.read_gpickle is not available
                with open(graph_path, "rb") as f:
                    G = pickle.load(f)

                # Extract basic graph information
                num_nodes = G.number_of_nodes()
                num_edges = G.number_of_edges()

                # Check if the graph is weighted and has labels
                is_weighted = False
                has_labels = False

                if G.number_of_nodes() > 0:
                    # Sample a node to check attributes
                    sample_node = next(iter(G.nodes(data=True)))
                    is_weighted = 'weight' in sample_node[1]
                    has_labels = 'label' in sample_node[1]

                # Create a test case dictionary
                test_case = {
                    'name': graph_path.stem,
                }

                # If the graph is labeled, extract the MIS solution and remove the labels from the graph
                if has_labels:
                    # Remove the label information from all nodes so the solver cannot use it
                    for node in G.nodes():
                        if 'label' in G.nodes[node]:
                            del G.nodes[node]['label']

                test_case['graph'] = G

                return test_case

            except Exception as e:
                raise Exception(f"Error loading graph from {graph_path}: {e}")

        # Handle single file or directory
        if file_path.is_file() and file_path.suffix == '.gpickle':
            test_case = process_graph_file(file_path)
            if test_case:
                test_cases.append(test_case)
        elif file_path.is_dir():
            for graph_path in file_path.rglob("*.gpickle"):
                test_case = process_graph_file(graph_path)
                if test_case:
                    test_cases.append(test_case)
        else:
            raise Exception(f"Invalid file path: {file_path}. Expected .gpickle file or directory.")

        return test_cases

    def eval_func(self, **kwargs):
        """
        Evaluate a Maximum Independent Set solution for correctness.
        Args:
            name (str): Name of the test case
            graph (networkx.Graph): The graph that was solved
            mis_nodes (list): List of nodes claimed to be in the maximum independent set
            mis_size (int): Claimed size of the maximum independent set
        Returns:
            actual_size (int): The actual size of the provided solution
            # dict: Evaluation results containing:
            #     - is_valid (bool): Whether the solution is a valid independent set
            #     - actual_size (int): The actual size of the provided solution
            #     - score (int): The score of the solution (0 if invalid, actual_size if valid)
            #     - error (str, optional): Error message if any constraint is violated
        """

        graph = kwargs['graph']
        mis_nodes = kwargs['mis_nodes']

        # Check if mis_nodes is a list
        if not isinstance(mis_nodes, list):
            raise Exception("mis_nodes must be a list")

        # Check if all nodes in mis_nodes exist in the graph
        node_set = set(graph.nodes())
        for node in mis_nodes:
            if node not in node_set:
                raise Exception(f"Node {node} in solution does not exist in graph")

        # Check for duplicates in mis_nodes
        if len(mis_nodes) != len(set(mis_nodes)):
            raise Exception("Duplicate nodes in solution")

        # Check if mis_size matches the length of mis_nodes
        actual_size = len(mis_nodes)

        # Most important: Check if it's an independent set (no edges between any nodes)
        for i in range(len(mis_nodes)):
            for j in range(i + 1, len(mis_nodes)):
                if graph.has_edge(mis_nodes[i], mis_nodes[j]):
                    raise Exception(f"Not an independent set: edge exists between {mis_nodes[i]} and {mis_nodes[j]}")

        return actual_size

    def norm_score(self, results):
        optimal_scores = {
            "er_large_test": [382] * 16,
            "er_test": [46] * 128,
            "er_valid": [46] * 100,
        }
        normed = {}
        for case, (scores, error_message) in results.items():
            if case not in optimal_scores:
                continue  # Skip if there's no optimal score defined.
            optimal_list = optimal_scores[case]
            normed_scores = []
            # Compute normalized score for each index.
            for idx, score in enumerate(scores):
                if isinstance(score, (int, float)):
                    normed_scores.append(score / optimal_list[idx])
                else:
                    normed_scores.append(score)
            normed[case] = (normed_scores, error_message)

        return normed

    def get_dev(self):
        dev = {'er_large_test': [1, 0, 8, 10, 6],
               'er_valid': [i for i in range(100)]}

        return dev




