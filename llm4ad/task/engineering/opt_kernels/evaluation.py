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
import re

from typing import Any

from llm4ad.base import Evaluation

__all__ = ['KernelEvaluation']

class KernelEvaluation(Evaluation):
    """Evaluator for car mountain problem."""

    def __init__(
            self,
            args,
            timeout_seconds=300, **kwargs
    ):
        operation_name = self._find_operation_name(cuda_code=args.cuda_code)
        task_description = self._make_task_description(operation_name, args)

        super().__init__(
            template_program=args.cuda_code,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.args = args
        self.func_code = args.func_code
        self.cuda_code = args.cuda_code
        self.gpu_type = args.GPU_TYPE
        self.cuda_version = args.CUDA_VER

    @staticmethod
    def _find_operation_name(cuda_code: str) -> str:
        pattern = r'm\.def\([^,]+,\s*&(\w+)'
        matches = re.findall(pattern, cuda_code)
        if matches:
            return matches[0]

    @staticmethod
    def _make_task_description(operation_name: str, args) -> str:
        return f"""
You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. 
Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. 
The kernel will be run on a {args.GPU_TYPE} GPU with CUDA {args.CUDA_VER}.
"""



    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        pass

