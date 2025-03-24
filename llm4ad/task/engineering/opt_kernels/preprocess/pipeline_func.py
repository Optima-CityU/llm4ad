from typing import Any

import torch
from .func_code_output import FuncCodeOutput
from .func_code_verify import FuncCodeVerify
from .cuda_code_output import CudaCodeOutput
from .cuda_code_verify import CudaCodeVerify

from llm4ad.tools.llm.llm_api_https import HttpsApi

def graceful_eval_cleanup(device: torch.device):
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

def convert_to_functional_code(llm_for_convert: HttpsApi, args, retry: int = 10) -> tuple[bool, str]:
    func_converter = FuncCodeOutput(llm_for_convert)
    func_verifier = FuncCodeVerify(res_path=args.res_path, keep_temp=args.keep_temp)

    convert_success = False
    func_code = None
    error_message = None
    convert_retry = 0
    while not convert_success:
        func_code = func_converter.get_code(args.code_content, func_code, error_message)
        convert_success, error_message = func_verifier.verify_func_code(args.code_content, func_code, args.device)
        graceful_eval_cleanup(args.device)
        if convert_success:
            return convert_success, func_code
        convert_retry += 1
        print("Conversion failed, retrying...")
        if convert_retry >= retry:
            return convert_success, func_code

def translate_into_CUDA_kernel(llm_for_translate: HttpsApi, args, retry: int = 10) -> tuple[Any, None] | tuple[
    Any, Any]:
    cuda_translator = CudaCodeOutput(llm_for_translate)
    cuda_verifier = CudaCodeVerify(res_path=args.res_path, keep_temp=args.keep_temp)

    translate_success = False
    cuda_code = None
    error_message = None
    translate_retry = 0
    while not translate_success:
        cuda_code = cuda_translator.get_code(args.func_code, cuda_code, error_message)
        result_dict, error_message = cuda_verifier.evaluate_cuda_code(args.func_code, cuda_code, args.code_operation, args.device)
        graceful_eval_cleanup(args.device)
        if error_message is None:
            return result_dict, error_message
        translate_retry += 1
        print("Translation failed, retrying...")
        if translate_retry >= retry:
            return result_dict, error_message