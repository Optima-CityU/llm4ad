import os
import time
import json
import argparse

from llm4ad.task.engineering.opt_kernels import KernelEvaluation
from llm4ad.task.engineering.opt_kernels.preprocess.pipeline_func import convert_to_functional_code

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

# use the absolute path to avoid the path error
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, 'KernelBench')
RES_PATH = os.path.join(ABS_PATH, 'Results')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on KernelBench')
    parser.add_argument('--CUDA_HOME', type=str, default="/usr/local/cuda", help='cuda home directory')
    parser.add_argument('--CUDA_VER', type=str, default="12.4", help='cuda version')
    parser.add_argument('--GPU_TYPE', type=str, default="H100", help='gpu type')
    parser.add_argument('--GPU_ARCH', type=str, default="9.0", help='gpu arch')
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument('--keep_temp', choices=[True, False], default=True, help='keep_temp')
    args = parser.parse_args()
    return args


def main(args):
    # args to dict
    args_dict = vars(args)
    with open(os.path.join(args.res_path, 'run_info.json'), "w") as f:
        json.dump(args_dict, f, indent=4)
    config_dict = {
        "host": "hk-api.gptbest.vip", "key": "sk-le1LLTBIQGMfP47XCb924e88919c456aB21eB5Af20E05632", "timeout": 200
    }

    # llm_for_func_convert = HttpsApi(model="o1-preview-2024-09-12", **config_dict)
    llm_for_func_convert = HttpsApi(model="gpt-3.5-turbo", **config_dict)
    convert_success, func_code = convert_to_functional_code(llm_for_func_convert, args, retry=100)
    if not convert_success:
        print("Conversion failed!")
        return


    # o1_preview = HttpsApi(model="o1-preview-2024-09-12", **config_dict)
    # llm = HttpsApi(host="hk-api.gptbest.vip",  # your host endpoint, e.g., api.openai.com, api.deepseek.com
    #                key="sk-le1LLTBIQGMfP47XCb924e88919c456aB21eB5Af20E05632",  # your key, e.g., sk-xxxxxxxxxx
    #                model="gpt-3.5-turbo",  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
    #                timeout=20)
    # temp_dir = os.path.join(args.res_path, 'temp')
    # task = KernelEvaluation(temp_dir)


if __name__ == '__main__':
    args = parse_args()
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    args.res_path = os.path.join(RES_PATH, time_stamp)
    os.makedirs(args.res_path, exist_ok=True)


    code_level = "level1"
    code_file_name = "1_Square_matrix_multiplication_.py"
    full_code_file_path = os.path.join(DATA_PATH, code_level, code_file_name)
    with open(full_code_file_path, 'r') as f:
        code_content = f.read()
    code_operation = code_file_name[:-3]
    args.code_operation = code_operation
    args.code_content = code_content

    main(args)