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
import os
import torch
import random
import tempfile
import importlib
import numpy as np
import torch.utils.cpp_extension as cpp_extension

from typing import Any

from llm4ad.base.opt_kernels.evaluate import Evaluation
from llm4ad.base.code import TextFunctionProgramConverter



__all__ = ['KernelEvaluation']



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
You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. 
Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. 
The kernel will be run on a {args.GPU_TYPE} GPU with CUDA {args.CUDA_VER}.

The Python function that you need to implement is:
{str(python_func)}

The CUDA kernel that you need to optimize is:
{args.cuda_code}
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
            *args,
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
            kernel_fn(*args)
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
            kernel_fn(*args, fn=cuda_fn.forward)
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
        program_str = """

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define VECTOR_WIDTH 4

// This kernel uses CUDA 12.4’s asynchronous copy (cp.async) instructions available on H100
// to prefetch tiles into shared memory. It keeps the original 4‐way unrolling and double buffering
// strategy, but replaces the blocking loads with asynchronous copies to overlap memory transfer with computation.
template <typename scalar_t>
__global__ void compute_async_shared_kernel(const scalar_t* __restrict__ A,
                                            const scalar_t* __restrict__ B,
                                            scalar_t* __restrict__ C,
                                            int N) {
  // Allocate two ping–pong shared memory buffers for A and B.
  __shared__ scalar_t sA[2][TILE_WIDTH][TILE_WIDTH + 2];
  __shared__ scalar_t sB[2][TILE_WIDTH][TILE_WIDTH + 2];

  const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  const int col = blockIdx.x * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
  scalar_t value[VECTOR_WIDTH] = {0};

  const int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
  int stage = 0;

  // Preload the first tile asynchronously using cp.async.
  int tile = 0;
  int a_col = tile * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
  if (row < N && (a_col + VECTOR_WIDTH - 1) < N) {
      scalar_t* dest = &sA[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
      const scalar_t* src = &A[row * N + a_col];
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], %2;\n"
          :
          : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
          : "memory");
  } else {
      // Fallback for out‐of‐range elements.
      scalar_t* dest = &sA[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
      for (int i = 0; i < VECTOR_WIDTH; i++) {
          int idx = a_col + i;
          dest[i] = (row < N && idx < N) ? A[row * N + idx] : 0;
      }
  }

  int b_row = tile * TILE_WIDTH + threadIdx.y;
  if (b_row < N && (col + VECTOR_WIDTH - 1) < N) {
      scalar_t* dest = &sB[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
      const scalar_t* src = &B[b_row * N + col];
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], %2;\n"
          :
          : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
          : "memory");
  } else {
      scalar_t* dest = &sB[stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
      for (int i = 0; i < VECTOR_WIDTH; i++) {
          int idx = col + i;
          dest[i] = (b_row < N && idx < N) ? B[b_row * N + idx] : 0;
      }
  }
  // Wait for asynchronous copies to complete.
  asm volatile("cp.async.wait_all;\n" : : : "memory");
  __syncthreads();

  // Main loop over tiles with double buffering.
  for (int t = 0; t < num_tiles; t++) {
      int next_stage = 1 - stage;
      if (t + 1 < num_tiles) {
          int next_tile = t + 1;
          int a_col_next = next_tile * TILE_WIDTH + threadIdx.x * VECTOR_WIDTH;
          if (row < N && (a_col_next + VECTOR_WIDTH - 1) < N) {
              scalar_t* dest = &sA[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
              const scalar_t* src = &A[row * N + a_col_next];
              asm volatile(
                  "cp.async.cg.shared.global [%0], [%1], %2;\n"
                  :
                  : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
                  : "memory");
          } else {
              scalar_t* dest = &sA[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
              for (int i = 0; i < VECTOR_WIDTH; i++) {
                  int idx = a_col_next + i;
                  dest[i] = (row < N && idx < N) ? A[row * N + idx] : 0;
              }
          }

          int b_row_next = next_tile * TILE_WIDTH + threadIdx.y;
          if (b_row_next < N && (col + VECTOR_WIDTH - 1) < N) {
              scalar_t* dest = &sB[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
              const scalar_t* src = &B[b_row_next * N + col];
              asm volatile(
                  "cp.async.cg.shared.global [%0], [%1], %2;\n"
                  :
                  : "r"(dest), "l"(src), "n"(VECTOR_WIDTH * sizeof(scalar_t))
                  : "memory");
          } else {
              scalar_t* dest = &sB[next_stage][threadIdx.y][threadIdx.x * VECTOR_WIDTH];
              for (int i = 0; i < VECTOR_WIDTH; i++) {
                  int idx = col + i;
                  dest[i] = (b_row_next < N && idx < N) ? B[b_row_next * N + idx] : 0;
              }
          }
      }

      asm volatile("cp.async.wait_all;\n" : : : "memory");
      __syncthreads();

      // Compute the current tile using 4-way unrolling.
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; k += 4) {
          scalar_t a0 = sA[stage][threadIdx.y][k];
          scalar_t a1 = sA[stage][threadIdx.y][k + 1];
          scalar_t a2 = sA[stage][threadIdx.y][k + 2];
          scalar_t a3 = sA[stage][threadIdx.y][k + 3];

          scalar_t b0[VECTOR_WIDTH], b1[VECTOR_WIDTH], b2[VECTOR_WIDTH], b3[VECTOR_WIDTH];
          for (int v = 0; v < VECTOR_WIDTH; ++v) {
              b0[v] = sB[stage][k][threadIdx.x * VECTOR_WIDTH + v];
              b1[v] = sB[stage][k + 1][threadIdx.x * VECTOR_WIDTH + v];
              b2[v] = sB[stage][k + 2][threadIdx.x * VECTOR_WIDTH + v];
              b3[v] = sB[stage][k + 3][threadIdx.x * VECTOR_WIDTH + v];
          }

          for (int v = 0; v < VECTOR_WIDTH; ++v) {
              value[v] = __fmaf_rn(a0, b0[v], value[v]);
              value[v] = __fmaf_rn(a1, b1[v], value[v]);
              value[v] = __fmaf_rn(a2, b2[v], value[v]);
              value[v] = __fmaf_rn(a3, b3[v], value[v]);
          }
      }

      __syncthreads();
      stage = next_stage;
  }

  // Write back the computed tile to global memory.
  if (row < N && col < N) {
      if (col + VECTOR_WIDTH <= N) {
          float4 out_val = {value[0], value[1], value[2], value[3]};
          *reinterpret_cast<float4*>(&C[row * N + col]) = out_val;
      } else {
          for (int v = 0; v < VECTOR_WIDTH; ++v) {
              if (col + v < N)
                  C[row * N + col + v] = value[v];
          }
      }
  }
}

torch::Tensor compute_async_shared_matmul_cuda(torch::Tensor A, torch::Tensor B) {
  const int N = A.size(0);
  A = A.contiguous();
  B = B.contiguous();
  auto C = torch::zeros({N, N}, A.options());

  const int threads_x = TILE_WIDTH / VECTOR_WIDTH;
  dim3 block_dim(threads_x, TILE_WIDTH);
  dim3 grid_dim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "compute_async_shared_matmul_cuda", ([&] {
      compute_async_shared_kernel<scalar_t><<<grid_dim, block_dim>>>(
          A.data_ptr<scalar_t>(),
          B.data_ptr<scalar_t>(),
          C.data_ptr<scalar_t>(),
          N);
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &compute_async_shared_matmul_cuda, "Async Shared Memory Matrix Multiplication CUDA forward");
  }
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_code_path = os.path.join(temp_dir, "cuda_code.cu")
            python_func_path = os.path.join(temp_dir, "func.py")
            cuda_func_path = os.path.join(temp_dir, "func_cu.py")
            KernelEvaluation.write_multiple_files_to_multiple_paths(
                [cuda_code_path, python_func_path, cuda_func_path], [program_str, self.func_code, self.func_code]
            )
            try:
                # os.environ["TORCH_USE_CUDA_DSA"] = "1"
                cuda_fn = cpp_extension.load(
                    name=self.operation_name,
                    sources=[cuda_code_path],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    with_cuda=True,
                    verbose=False
                )
            except Exception as e:
                return None
            func_module, func_spec = KernelEvaluation.load_module_from_path(python_func_path, "func_module")
            cuda_module, cuda_spec = KernelEvaluation.load_module_from_path(cuda_func_path, "cu_module")

            KernelEvaluation.set_seed(0)
            init_inputs = func_module.get_init_inputs()
            init_inputs = [
                x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x for x in init_inputs
            ]

            # load model
            with torch.no_grad():
                KernelEvaluation.set_seed(0)
                original_model = func_module.Model(*init_inputs)
                try:
                    KernelEvaluation.set_seed(0)
                    custom_model = cuda_module.Model(*init_inputs)
                except Exception as e:
                    return None

            # check correctness
            torch.manual_seed(0)
            correctness_trial_seeds = [
                torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(10)
            ]
            with torch.no_grad():
                for trial in range(10):

                    trial_seed = correctness_trial_seeds[trial]

                    KernelEvaluation.set_seed(trial_seed)
                    inputs = func_module.get_inputs()
                    inputs = [
                        x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]

                    KernelEvaluation.set_seed(trial_seed)
                    model = original_model.cuda(device=self.device)

                    KernelEvaluation.set_seed(trial_seed)
                    model_new = custom_model.cuda(device=self.device)

                    output = model(*inputs)
                    torch.cuda.synchronize(device=self.device)
                    # ensure all GPU operations are completed before checking results

                    try:
                        output_new = model_new(*inputs, fn=cuda_fn.forward)
                        torch.cuda.synchronize(device=self.device)
                        if output.shape != output_new.shape:
                            return None
                        if not torch.allclose(output, output_new, atol=1e-02, rtol=1e-02):
                            return None
                    except Exception as e:
                        return None

            # test performance
            torch.cuda.synchronize(device=self.device)
            KernelEvaluation.set_seed(0)
            inputs = func_module.get_inputs()
            inputs = [
                x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=self.device)
            torch.cuda.synchronize(device=self.device)

            elapsed_times = KernelEvaluation.time_execution_with_cuda_event(
                model_new,
                *inputs,
                cuda_fn,
                num_trials=10,
                verbose=False,
                device=self.device,
            )
            runtime_stats = np.mean(elapsed_times)
            return -runtime_stats

