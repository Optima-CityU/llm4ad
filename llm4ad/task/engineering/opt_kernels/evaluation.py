# TODO: Adding descriptions
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

from llm4ad.base import Evaluation

__all__ = ['KernelEvaluation']

class KernelEvaluation(Evaluation):
    """Evaluator for car mountain problem."""

    def __init__(
            self,
            func_code: str,
            cuda_code: str,
            timeout_seconds=300, **kwargs
    ):

        super().__init__(
            template_program="",
            task_description="task_description",
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.func_code = func_code
        self.cuda_code = cuda_code

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        pass
