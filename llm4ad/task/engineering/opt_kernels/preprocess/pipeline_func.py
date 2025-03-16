import torch
from .func_code_output import FuncCodeOutput
from .func_code_verify import FuncCodeVerify

from llm4ad.tools.llm.llm_api_https import HttpsApi

def graceful_eval_cleanup(device: torch.device):
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

def convert_to_functional_code(llm_for_convert: HttpsApi, args, retry: int = 50) -> tuple[bool, str]:
    func_converter = FuncCodeOutput(llm_for_convert)
    func_verifier = FuncCodeVerify(res_path=args.res_path)

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
        if convert_retry >= retry:
            return convert_success, func_code
    #     if not convert_success:
    #         print("Conversion failed, retrying...")
    # print("Conversion successful!")