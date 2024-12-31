from __future__ import annotations

import multiprocessing
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, List

from .code import TextFunctionProgramConverter, Program
from .modify_code import ModifyCode


class MultiProgramEvaluator(ABC):
    def __init__(
            self,
            use_numba_accelerate: bool = False,
            use_protected_div: bool = False,
            protected_div_delta: float = 1e-5,
            random_seed: int | None = None,
            timeout_seconds: int | float = None,
            *,
            exec_code: bool = True,
            safe_evaluate: bool = True,
            daemon_eval_process: bool = False
    ):
        """Evaluator for executing generated code.
        Args:
            use_numba_accelerate: Wrap the function with '@numba.jit(nopython=True)'.
            use_protected_div   : Modify 'a / b' => 'a / (b + delta)'.
            protected_div_delta : Delta value in protected div.
            random_seed         : If is not None, set random seed in the first line of the function body.
            timeout_seconds     : Terminate the evaluation after timeout seconds.
            exec_code           : Using 'exec()' to compile the code and provide the callable function.
                If is set to 'False', the 'callable_func' argument in 'self.evaluate_program' is always 'None'.
                If is set to 'False', the user should provide the score of the program based on 'program_str' argument in 'self.evaluate_program'.
            safe_evaluate       : Evaluate in safe mode using a new process. If is set to False,
                the evaluation will not be terminated after timeout seconds. The user should consider how to
                terminate evaluating in time.
            daemon_eval_process : Set the evaluate process as a daemon process. If set to True,
                you can not set new processes in the evaluator. Which means in self.evaluate_program(),
                you can not create new processes.

        -Assume that: use_numba_accelerate=True, self.use_protected_div=True, and self.random_seed=2024.
        -The original function:
        --------------------------------------------------------------------------------
        import numpy as np

        def f(a, b):
            a = np.random.random()
            return a / b
        --------------------------------------------------------------------------------
        -The modified function will be:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f():
            np.random.seed(2024)
            a = np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator, and '_protected_dev' will be added by this function.
        """
        self.use_numba_accelerate = use_numba_accelerate
        self.use_protected_div = use_protected_div
        self.protected_div_delta = protected_div_delta
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.exec_code = exec_code
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process

    @abstractmethod
    def evaluate_programs(self, program_strs: List[str], callable_funcs: List[callable], **kwargs) -> Any | None:
        r"""Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_strs  : The programs in string. You can _ignore this argument when implementation.
            callable_funcs: The callable heuristic functions to be eval.
        Return:
            Returns the fitness value.
        """
        raise NotImplementedError('Must provide a evaluator for a function.')


class MultiProgramSecureEvaluator:
    """A more generic version of `llm4ad.base.evaluate.SecureEvaluator`, which can evaluate more than one candidate heuristics.
    `MultiProgramSecureEvaluator` evaluates `llm4ad.base.evaluate_multi_program.MultiProgramEvaluator` in safe process.
    """

    def __init__(self, multi_program_evaluator: MultiProgramEvaluator, debug_mode=False, *, fork_proc: str | bool = 'auto'):
        assert fork_proc in [True, False, 'auto', 'default']
        self._multi_program_evaluator = multi_program_evaluator
        self._debug_mode = debug_mode

        if self._multi_program_evaluator.safe_evaluate:
            if fork_proc == 'auto':
                # force MacOS and Linux use 'fork' to generate new process
                if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                    multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is True:
                multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is False:
                multiprocessing.set_start_method('spawn', force=True)

    def _modify_program_code(self, program_str: str) -> str:
        function_name = TextFunctionProgramConverter.text_to_function(program_str).name
        if self._multi_program_evaluator.use_numba_accelerate:
            program_str = ModifyCode.add_numba_decorator(
                program_str, function_name=function_name
            )
        if self._multi_program_evaluator.use_protected_div:
            program_str = ModifyCode.replace_div_with_protected_div(
                program_str, self._multi_program_evaluator.protected_div_delta, self._multi_program_evaluator.use_numba_accelerate
            )
        if self._multi_program_evaluator.random_seed is not None:
            program_str = ModifyCode.add_numpy_random_seed_to_func(
                program_str, function_name, self._multi_program_evaluator.random_seed
            )
        return program_str

    def evaluate_programs(self, programs: List[str] | List[Program], **kwargs):
        try:
            program_strs = [str(p) for p in programs]
            # record function name BEFORE modifying program code
            function_names = [TextFunctionProgramConverter.text_to_function(program_str).name
                              for program_str in program_strs]
            program_strs = [self._modify_program_code(program_str)
                            for program_str in program_strs]

            if self._debug_mode:
                print(f'DEBUG: evaluated program:\n')
                for program_str in program_strs:
                    print(f'{program_str}\n')

            # safe evaluate
            if self._multi_program_evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process,
                    args=(program_strs, function_names, result_queue),
                    kwargs=kwargs,
                    daemon=self._multi_program_evaluator.daemon_eval_process
                )
                process.start()

                if self._multi_program_evaluator.timeout_seconds is not None:
                    try:
                        # get the result in timeout seconds
                        result = result_queue.get(timeout=self._multi_program_evaluator.timeout_seconds)
                        # after getting the result, terminate/kill the process
                        process.terminate()
                        process.join()
                    except:
                        # timeout
                        if self._debug_mode:
                            print(f'DEBUG: the evaluation time exceeds {self._multi_program_evaluator.timeout_seconds}s.')
                        process.terminate()
                        process.join()
                        result = None
                else:
                    result = result_queue.get()
                    process.terminate()
                    process.join()
                # return evaluate result
                return result
            else:
                return self._evaluate(program_strs, function_names)
        except Exception as e:
            if self._debug_mode:
                print(f'In evaluate_programs_record_time(), err: {e}')
            # return None and evaluate time
            return None

    def evaluate_programs_record_time(self, programs: List[str] | List[Program], **kwargs):
        evaluate_start = time.time()
        result = self.evaluate_programs(programs, **kwargs)
        return result, time.time() - evaluate_start

    def _evaluate_in_safe_process(self, program_strs: List[str], function_names: List[str], result_queue: multiprocessing.Queue, **kwargs):
        try:
            if self._multi_program_evaluator.exec_code:
                # compile the program, and maps the global func/var/class name to its address
                all_globals_namespace = {}
                # execute the program, map func/var/class to global namespace
                for program_str in program_strs:
                    exec(program_str, all_globals_namespace)
                # get the pointer of 'function_to_run'
                program_callables = [all_globals_namespace[function_name]
                                     for function_name in function_names]
            else:
                program_callables = None
            # get evaluate result
            res = self._multi_program_evaluator.evaluate_programs(program_strs, program_callables, **kwargs)
            result_queue.put(res)
        except Exception as e:
            if self._debug_mode:
                print(f'In _evaluate_in_safe_process(), err: {e}')
            result_queue.put(None)

    def _evaluate(self, program_strs: List[str], function_names: List[str], **kwargs):
        try:
            if self._multi_program_evaluator.exec_code:
                # compile the program, and maps the global func/var/class name to its address
                all_globals_namespace = {}
                # execute the program, map func/var/class to global namespace
                for program_str in program_strs:
                    exec(program_str, all_globals_namespace)
                # get the pointer of 'function_to_run'
                program_callables = [all_globals_namespace[function_name]
                                     for function_name in function_names]
            else:
                program_callables = None
            # get evaluate result
            res = self._multi_program_evaluator.evaluate_programs(program_strs, program_callables, **kwargs)
            return res
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None
