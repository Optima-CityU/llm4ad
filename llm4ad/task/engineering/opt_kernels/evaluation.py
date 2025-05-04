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

import logging
import re
import os
import sys
import time
import torch
import random
import tempfile
import importlib
import numpy as np
import torch.utils.cpp_extension as cpp_extension
import torch.multiprocessing as mp
from torch.utils.benchmark import Timer
from torch.profiler import profile, record_function, ProfilerActivity

from typing import Any

from llm4ad.base.opt_kernels.evaluate import Evaluation
from llm4ad.base.code import TextFunctionProgramConverter

__all__ = ['KernelEvaluation']

def check_correctness(func_model_inst, func_model_inst_copy, get_inputs, cuda_fn, atol=1e-02, rtol=1e-02):
    with torch.no_grad():
        inputs = get_inputs()
        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        model = func_model_inst.cuda()
        model_new = func_model_inst_copy.cuda()

        output = model(*inputs)
        torch.cuda.synchronize()
        try:
            output_new = model_new(*inputs, fn=cuda_fn.forward)
            torch.cuda.synchronize()
            if output.shape != output_new.shape:
                return False, f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
            if not torch.allclose(output, output_new, atol=atol, rtol=rtol):
                max_diff = torch.max(torch.abs(output - output_new)).item()
                avg_diff = torch.mean(torch.abs(output - output_new)).item()
                return False, f"Output mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}"
        except Exception as e:
            return False, f"Error running CUDA code: {e}"
    return True, None


def evaluate_cuda_code(func_torch_code: str, code_operation: str, res_path: str, seed: int = 0, test_mode="kernelbench"):
    cuda_fname = os.path.join(res_path, "test_cuda_code.cu")
    build_dir = os.path.join(res_path, "build")
    os.makedirs(build_dir, exist_ok=True)

    try:
        # os.environ["TORCH_USE_CUDA_DSA"] = "1"
        cuda_fn = cpp_extension.load(
            name=code_operation,
            sources=[cuda_fname],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_dir,
            with_cuda=True,
            verbose=False
        )
    except Exception as e:
        # safe_reset_cuda()
        return dict(), f"Error loading CUDA code: {e}"
    func_code_path = os.path.join(res_path, "func.py")
    func_code_copy_path = os.path.join(res_path, "func_copy.py")
    KernelEvaluation.write_multiple_files_to_multiple_paths([func_code_path, func_code_copy_path], [func_torch_code, func_torch_code])

    func_module, func_spec = KernelEvaluation.load_module_from_path(func_code_path, "func_module")
    func_module_copy, func_spec_copy = KernelEvaluation.load_module_from_path(func_code_copy_path, "func_module_copy")

    init_inputs = func_module.get_init_inputs()
    init_inputs = [
        x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        KernelEvaluation.set_seed(seed)
        func_model_inst = func_module.Model(*init_inputs)
        KernelEvaluation.set_seed(seed)
        func_model_inst_copy = func_module_copy.Model(*init_inputs)

    for i in range(5):
        try:
            correctness, error_message = check_correctness(
                func_model_inst, func_model_inst_copy, func_module.get_inputs, cuda_fn
            )
        except Exception as e:
            return dict(), f"Error running CUDA code: {e}"
        if not correctness:
            return dict(), error_message

    del func_model_inst
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    func_model_inst_copy = func_model_inst_copy.cuda()
    # Now that the CUDA code is correct, we can evaluate the performance
    with torch.no_grad():
        inputs = func_module.get_inputs()
        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
    if test_mode == "sakana":
        cuda_timer = Timer(
            stmt="model(*inputs, fn=cuda_fn.forward)",
            globals={
                "model": func_model_inst_copy.cuda(),
                "inputs": inputs,
                "cuda_fn": cuda_fn,
            },
        )
        cuda_runtime = cuda_timer.timeit(100).mean * 1000
    else:
        func_model_inst_copy.cuda()
        run_time_list = []
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            func_model_inst_copy(*inputs, fn=cuda_fn.forward)
            end_event.record()

            # Synchronize to ensure the events have completed
            torch.cuda.synchronize()

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)

            run_time_list.append(elapsed_time_ms)
        cuda_runtime = sum(run_time_list) / 100.0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            func_model_inst_copy(*inputs, fn=cuda_fn.forward)
    prof_string = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

    return dict(cuda_runtime=cuda_runtime, prof_string=prof_string), None


class KernelEvaluation(Evaluation):
    """Evaluator for car mountain problem."""

    def __init__(
            self,
            args,
            timeout_seconds=300, **kwargs
    ):
        python_program = TextFunctionProgramConverter.text_to_program(args.func_code)
        for each_python_func in python_program.functions:
            if each_python_func.name == 'module_fn':
                python_func = each_python_func
                break
        self.python_func = python_func
        self.operation_name = self._find_operation_name(cuda_code=args.cuda_code)
        task_description = self._make_task_description(self.operation_name, args, python_func)

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
        self.device = args.device

    @staticmethod
    def _find_operation_name(cuda_code: str) -> str:
        pattern = r'm\.def\([^,]+,\s*&(\w+)'
        matches = re.findall(pattern, cuda_code)
        if matches:
            return matches[0]

    @staticmethod
    def _make_task_description(operation_name: str, args, python_func) -> str:
        return f"""
You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a {args.GPU_TYPE} GPU with CUDA {args.CUDA_VER}.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the CUDA kernel code example you need to optimize:
```cpp
{args.cuda_code}
```
"""

    @staticmethod
    def load_module_from_path(code_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, spec

    @staticmethod
    def write_multiple_files_to_multiple_paths(file_paths, contents):
        for file_path, content in zip(file_paths, contents):
            KernelEvaluation.write_file_to_path(file_path, content)

    @staticmethod
    def write_file_to_path(file_path, content):
        with open(file_path, "w") as f:
            f.write(content)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def time_execution_with_cuda_event(
            kernel_fn: callable,
            inputs,
            cuda_fn,
            num_warmup: int = 3,
            num_trials: int = 10,
            verbose: bool = True,
            device: torch.device = None,
    ) -> list[float]:
        """
        Time a CUDA kernel function over multiple trials using torch.cuda.Event

        Args:
            kernel_fn: Function to time
            *args: Arguments to pass to kernel_fn
            num_trials: Number of timing trials to run
            verbose: Whether to print per-trial timing info
            device: CUDA device to use, if None, use current device

        Returns:
            List of elapsed times in milliseconds
        """
        if device is None:
            if verbose:
                print(f"Using current device: {torch.cuda.current_device()}")
            device = torch.cuda.current_device()

        # Warm ups
        for _ in range(num_warmup):
            kernel_fn(*inputs)
            torch.cuda.synchronize(device=device)

        print(
            f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
        )
        elapsed_times = []

        # Actual trials
        for trial in range(num_trials):
            # create event marker default is not interprocess
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            kernel_fn(*inputs, fn=cuda_fn.forward)
            end_event.record()

            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=device)

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            if verbose:
                print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

        return elapsed_times

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        temp_str = tempfile.mktemp().split('/')[-1]
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = os.path.join(self.args.res_path, "all_samples", f"{time_stamp}-{temp_str}")
        os.makedirs(temp_dir, exist_ok=True)

        cuda_code_path = os.path.join(temp_dir, "test_cuda_code.cu")
        python_func_path = os.path.join(temp_dir, "func.py")
        cuda_func_path = os.path.join(temp_dir, "func_cu.py")
        KernelEvaluation.write_multiple_files_to_multiple_paths(
            [cuda_code_path, python_func_path, cuda_func_path], [program_str, self.func_code, self.func_code]
        )
        build_dir = os.path.join(temp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        try:
            # os.environ["TORCH_USE_CUDA_DSA"] = "1"
            cuda_fn = cpp_extension.load(
                name=f"{self.operation_name}_{temp_str}",
                sources=[cuda_code_path],
                build_directory=build_dir,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=False
            )
        except Exception as e:
            self.save_cuda_code_and_error(program_str, e)
            return None


        self.args.mutex.acquire()
        if sys.platform.startswith('linux'):
            mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=1) as pool:
            res = pool.apply_async(evaluate_cuda_code, (self.func_code, f"{self.operation_name}_{temp_str}", temp_dir, 0, "kernelbench"))
            try:
                result = res.get(timeout=self.timeout_seconds)
                if "cuda_runtime" in result[0]:
                    return -result[0]["cuda_runtime"]
                else:
                    self.save_cuda_code_and_error(program_str, result[1])
                    return None

            except Exception as e:
                result = dict(cuda_runtime=np.inf, prof_string="Error"), f"{e}"
                self.save_cuda_code_and_error(program_str, e)
                return -result[0]["cuda_runtime"]
            finally:
                self.args.mutex.release()

        # try:
        #     func_module, func_spec = KernelEvaluation.load_module_from_path(python_func_path, "func_module")
        #     cuda_module, cuda_spec = KernelEvaluation.load_module_from_path(cuda_func_path, "cu_module")
        #
        #     KernelEvaluation.set_seed(0)
        #     init_inputs = func_module.get_init_inputs()
        #     init_inputs = [
        #         x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        #     ]
        #     # load model
        #     with torch.no_grad():
        #         KernelEvaluation.set_seed(0)
        #         original_model = func_module.Model(*init_inputs)
        #         try:
        #             KernelEvaluation.set_seed(0)
        #             custom_model = cuda_module.Model(*init_inputs)
        #         except Exception as e:
        #             self.save_cuda_code_and_error(program_str, e)
        #
        #             return None
        #
        #     # check correctness
        #     torch.manual_seed(0)
        #     correctness_trial_seeds = [
        #         torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(10)
        #     ]
        #     with torch.no_grad():
        #         for trial in range(10):
        #
        #             trial_seed = correctness_trial_seeds[trial]
        #
        #             KernelEvaluation.set_seed(trial_seed)
        #             inputs = func_module.get_inputs()
        #             inputs = [
        #                 x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
        #                 for x in inputs
        #             ]
        #
        #             KernelEvaluation.set_seed(trial_seed)
        #             model = original_model.cuda(device=self.device)
        #
        #             KernelEvaluation.set_seed(trial_seed)
        #             model_new = custom_model.cuda(device=self.device)
        #
        #             output = model(*inputs)
        #             torch.cuda.synchronize(device=self.device)
        #             # ensure all GPU operations are completed before checking results
        #
        #             try:
        #                 output_new = model_new(*inputs, fn=cuda_fn.forward)
        #                 torch.cuda.synchronize(device=self.device)
        #                 if output.shape != output_new.shape:
        #                     self.save_cuda_code_and_error(program_str, "output not match")
        #                     return None
        #                 if not torch.allclose(output, output_new, atol=1e-02, rtol=1e-02):
        #                     self.save_cuda_code_and_error(program_str, "results not match")
        #                     return None
        #             except Exception as e:
        #                 self.save_cuda_code_and_error(program_str, e)
        #                 return None
        #
        #     # test performance
        #     torch.cuda.synchronize(device=self.device)
        #     KernelEvaluation.set_seed(0)
        #     inputs = func_module.get_inputs()
        #     inputs = [
        #         x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
        #         for x in inputs
        #     ]
        #     model_new = custom_model.cuda(device=self.device)
        #     torch.cuda.synchronize(device=self.device)
        #
        #     elapsed_times = KernelEvaluation.time_execution_with_cuda_event(
        #         model_new,
        #         inputs,
        #         cuda_fn,
        #         num_trials=10,
        #         verbose=False,
        #         device=self.device,
        #     )
        #     runtime_stats = np.mean(elapsed_times)
        #     return -runtime_stats
        # except Exception as e:
        #     self.save_cuda_code_and_error(program_str, e)
        # finally:
        #     self.args.mutex.release()

    def save_cuda_code_and_error(self, cuda_code, error_info):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        error_dir = os.path.join(self.args.res_path, f"error_{time_stamp}")
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, f"cuda_code.cu"), "w") as f:
            f.write(cuda_code)
        with open(os.path.join(error_dir, f"error_info.txt"), "w") as f:
            if type(error_info) is not str:
                f.write(str(error_info))
            else:
                f.write(error_info)
