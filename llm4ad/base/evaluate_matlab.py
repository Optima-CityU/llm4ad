# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
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

import multiprocessing
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

import matlab.engine

from .code_matlab import TextMatlabFunctionProgramConverter, MatlabProgram
from .modify_code_matlab import ModifyMatlabCode


class MatlabEvaluation(ABC):
    def __init__(
            self,
            template_program: str | MatlabProgram,
            task_description: str = '',
            use_protected_div: bool = False,
            protected_div_delta: float = 1e-5,
            random_seed: int | None = None,
            timeout_seconds: int | float = None,
            *,
            safe_evaluate: bool = True,
            daemon_eval_process: bool = False
    ):
        """Evaluation interface for executing generated MATLAB code.
        Args:
            use_protected_div   : Modify 'a / b' => 'a ./ (b + delta)'.
            protected_div_delta : Delta value in protected div.
            random_seed        : If is not None, set random seed using rng(seed).
            timeout_seconds    : Terminate the evaluation after timeout seconds.
            matlab_path       : Path to MATLAB installation. If None, uses default.
            safe_evaluate     : Evaluate in safe mode using a new process.
            daemon_eval_process: Set the evaluate process as a daemon process.

        -Assume that: self.use_protected_div=True, and self.random_seed=2024.
        -The original function:
        --------------------------------------------------------------------------------
        function y = f(a, b)
            a = rand();
            y = a / b;
        end
        --------------------------------------------------------------------------------
        -The modified function will be:
        --------------------------------------------------------------------------------
        function y = f(a, b)
            rng(2024);
            a = rand();
            y = protected_div(a, b);
        end

        function y = protected_div(x, y, delta)
            if nargin < 3
                delta = 1e-5;
            end
            y = x ./ (y + delta);
        end
        --------------------------------------------------------------------------------
        """
        self.template_program = template_program
        self.task_description = task_description
        self.use_protected_div = use_protected_div
        self.protected_div_delta = protected_div_delta
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process

    @abstractmethod
    def evaluate_program(self, program_str: str, **kwargs) -> Any | None:
        """Evaluate a given MATLAB function using the MATLAB engine.
        Args:
            program_str: The MATLAB function in string format.
        Return:
            Returns the fitness value.
        """
        raise NotImplementedError('Must provide a evaluator for a MATLAB function.')


class SecureMatlabEvaluator:
    def __init__(
            self,
            evaluator: MatlabEvaluation,
            debug_mode=False,
            *,
            fork_proc: Literal['auto', 'default'] | bool = 'auto',
            **kwargs
    ):
        assert fork_proc in [True, False, 'auto', 'default']
        self._evaluator = evaluator
        self._debug_mode = debug_mode

        if self._evaluator.safe_evaluate:
            if fork_proc == 'auto':
                if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                    multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is True:
                multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is False:
                multiprocessing.set_start_method('spawn', force=True)

    def _modify_program_code(self, program_str: str) -> str:
        """Modify the MATLAB program code according to configuration."""
        function_name = TextMatlabFunctionProgramConverter.text_to_function(program_str).name

        if self._evaluator.use_protected_div:
            program_str = ModifyMatlabCode.replace_div_with_protected_div(
                program_str, self._evaluator.protected_div_delta
            )

        if self._evaluator.random_seed is not None:
            program_str = ModifyMatlabCode.add_rng_seed_to_func(
                program_str, function_name, self._evaluator.random_seed
            )

        return program_str

    def evaluate_program(self, program: str | MatlabProgram, **kwargs):
        try:
            program_str = str(program)
            # record function name BEFORE modifying program code
            function_name = TextMatlabFunctionProgramConverter.text_to_function(program_str).name

            program_str = self._modify_program_code(program_str)
            if self._debug_mode:
                print(f'DEBUG: evaluated program:\n{program_str}\n')

            # safe evaluate
            if self._evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process,
                    args=(program_str, function_name, result_queue),
                    kwargs=kwargs,
                    daemon=self._evaluator.daemon_eval_process
                )
                process.start()

                if self._evaluator.timeout_seconds is not None:
                    try:
                        # get the result in timeout seconds
                        result = result_queue.get(timeout=self._evaluator.timeout_seconds)
                        # after getting the result, terminate/kill the process
                        process.terminate()
                        process.join()
                    except Exception as e:
                        # timeout
                        if self._debug_mode:
                            print(f'DEBUG: the evaluation time exceeds {self._evaluator.timeout_seconds}s.')
                        process.terminate()
                        process.join()
                        result = None
                else:
                    result = result_queue.get()
                    process.terminate()
                    process.join()
                return result
            else:
                return self._evaluate(program_str, function_name, **kwargs)
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None

    def evaluate_program_record_time(self, program: str | MatlabProgram, **kwargs):
        evaluate_start = time.time()
        result = self.evaluate_program(program, **kwargs)
        return result, time.time() - evaluate_start

    def _evaluate_in_safe_process(self, program_str: str, function_name, result_queue: multiprocessing.Queue, **kwargs):
        try:
            # get evaluate result
            res = self._evaluator.evaluate_program(program_str, **kwargs)
            result_queue.put(res)
        except Exception as e:
            if self._debug_mode:
                print(e)
            result_queue.put(None)

    def _evaluate(self, program_str: str, function_name, **kwargs):
        try:
            # get evaluate result
            res = self._evaluator.evaluate_program(program_str, **kwargs)
            return res
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None
