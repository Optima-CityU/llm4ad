# Module Name: TSPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the constructive heuristic for Traveling Salseman Problem (TSP).
#              Given a set of locations,
#              the goal is to find optimal route to travel all locations and back to start point
#              while minimizing the total travel distance.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - problem_size: Number of customers to serve: int (default: 50).
#
# 
# References:
#   - Fei Liu, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. 
#     "Algorithm Evolution using Large Language Model." arXiv preprint arXiv:2311.15249 (2023).
# 
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any
import numpy as np
import os
import matlab.engine
from llm4ad.base import MatlabEvaluation
from llm4ad.task.optimization.tsp_construct_matlab.template import template_program, task_description

__all__ = ['TSPEvaluationMat']


class TSPEvaluationMat(MatlabEvaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=500,
                 n_instance=16,
                 problem_size=50,
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
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.problem_size = problem_size

        # current_file = os.path.abspath(__file__)
        # project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        # matlab_src_path = os.path.join(project_root, 'optimization/tsp_construct_matlab/matlab_src')
        # self.eng = matlab.engine.start_matlab(f'-sd "{matlab_src_path}"')

    def evaluate_program(self, program_str: str) -> Any | None:
        return self.evaluate(program_str)

    def evaluate(self, program_str: str) -> float:
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        matlab_src_path = os.path.join(project_root, 'optimization/tsp_construct_matlab/matlab_src')
        eng = matlab.engine.start_matlab(f'-sd "{matlab_src_path}"')

        # write the program to tsp_construct.m
        with open(f'{matlab_src_path}/tsp_construct.m', 'w') as f:
            f.write(program_str)

        avg_fit, best_route, best_dist, best_matrix = eng.evaluate(self.problem_size, self.n_instance, nargout=4)

        eng.quit()

        return -avg_fit
