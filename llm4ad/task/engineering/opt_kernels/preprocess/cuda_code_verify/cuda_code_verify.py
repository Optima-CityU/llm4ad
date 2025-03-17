import os
import sys
import time
import torch
import shutil
import tempfile




import torch.multiprocessing as mp
import torch.utils.cpp_extension as cpp_extension
from torch.utils.benchmark import Timer
from torch.profiler import profile, record_function, ProfilerActivity

from ..code_verify import CodeVerify

def compile_cuda_code(code_operation: str, cuda_fname: str, build_dir: str):
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        cuda_fn = cpp_extension.load(
            name=code_operation,
            sources=[cuda_fname],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_dir,
            with_cuda=True,
            verbose=False
        )
        return cuda_fn, None
    except Exception as e:
        # safe_reset_cuda()
        return None, f"Error loading CUDA code: {e}"

def evaluate_cuda_code(org_torch_code: str, func_torch_code: str, cuda_code: str, code_operation: str, res_path: str, device: torch.device):
    # get cpu count
    cpu_count = os.cpu_count()
    os.environ['MAX_JOBS'] = f"{cpu_count}"
    cuda_fname = os.path.join(res_path, "test_cuda_code.cu")
    CudaCodeVerify.write_file_to_path(cuda_fname, cuda_code)
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
    org_code_path = os.path.join(res_path, "original.py")
    func_code_path = os.path.join(res_path, "func.py")
    func_code_copy_path = os.path.join(res_path, "func_copy.py")
    CudaCodeVerify.write_multiple_files_to_multiple_paths([org_code_path, func_code_path, func_code_copy_path], [org_torch_code, func_torch_code, func_torch_code])

    org_module, org_spec = CudaCodeVerify.load_module_from_path(org_code_path, "org_module")
    func_module, func_spec = CudaCodeVerify.load_module_from_path(func_code_path, "func_module")
    func_module_copy, func_spec_copy = CudaCodeVerify.load_module_from_path(func_code_copy_path, "func_module_copy")

    init_inputs = org_module.get_init_inputs()
    init_inputs = [
        x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        func_model_inst = func_module.Model(*init_inputs)
        func_model_inst_copy = func_module_copy.Model(*init_inputs)

    for i in range(5):
        try:
            correctness, error_message = check_correctness(
                func_model_inst, func_model_inst_copy, org_module.get_inputs, cuda_fn
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
        inputs = org_module.get_inputs()
        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
    cuda_timer = Timer(
        stmt="model(*inputs, fn=cuda_fn.forward)",
        globals={
            "model": func_model_inst_copy.cuda(),
            "inputs": inputs,
            "cuda_fn": cuda_fn,
        },
    )
    cuda_runtime = cuda_timer.timeit(100).mean * 1000
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            func_model_inst_copy(*inputs, fn=cuda_fn.forward)
    prof_string = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

    return dict(cuda_runtime=cuda_runtime, prof_string=prof_string), None

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


class CudaCodeVerify(CodeVerify):
    def __init__(self, verify_counts=5, res_path=None, keep_temp=False):
        super().__init__(verify_counts=verify_counts, res_path=res_path, keep_temp=keep_temp, specifier="CudaCodeVerify")

    def evaluate_cuda_code(self, org_torch_code: str, func_torch_code: str, cuda_code: str, code_operation: str, device: torch.device):
        if sys.platform.startswith('linux'):
            mp.set_start_method("spawn", force=True)

        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.TemporaryDirectory(dir=self.res_path, prefix=f"{time_stamp}_")
        os.makedirs(temp_dir.name, exist_ok=True)
        if os.path.exists(os.path.join(temp_dir.name, "build")):
            shutil.rmtree(os.path.join(temp_dir.name, "build"))

        with mp.Pool(processes=1) as pool:
            res = pool.apply_async(evaluate_cuda_code, (org_torch_code, func_torch_code, cuda_code, code_operation, temp_dir.name, device))
            try:
                result = res.get(timeout=300)
            except Exception as e:
                result = dict(), f"Timeout compiling CUDA code."

        shutil.rmtree(os.path.join(temp_dir.name, "build"))
        return result

    def get_time_torch(self, torch_code: str):
        torch_fname = os.path.join(self.res_path, "test_torch_code.py")
        with open(torch_fname, "w") as f:
            f.write(torch_code)

        task, spec = self.load_module_from_path(os.path.join(torch_fname), "test_torch_code")
        init_inputs = task.get_init_inputs()
        model = task.Model(*init_inputs)
        with torch.no_grad():
            inputs = task.get_inputs()
            inputs = [
                x.cuda() if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

        torch_timer = Timer(
            stmt="model(*inputs)",
            globals={
                "model": model.cuda(),
                "inputs": inputs
            },
        )
        torch_runtime = torch_timer.timeit(100).mean * 1000
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return torch_runtime
