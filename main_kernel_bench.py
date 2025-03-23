import os
import time
import argparse

# use the absolute path to avoid the path error
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(ABS_PATH, 'Results')
DATA_PATH = os.path.join(ABS_PATH, 'CUDABench', 'level1', '1_Square_matrix_multiplication_')

from llm4ad.task.engineering.opt_kernels import KernelEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on KernelBench')
    # My local computer
    # parser.add_argument('--CUDA_HOME', type=str, default="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin", help='cuda home directory')
    # parser.add_argument('--CUDA_VER', type=str, default="12.6", help='cuda version')
    # parser.add_argument('--GPU_TYPE', type=str, default="RTX 4060 Ti", help='gpu type')
    # parser.add_argument('--GPU_ARCH', type=str, default="8.9", help='gpu arch')
    # computer of CityU
    parser.add_argument('--CUDA_HOME', type=str,
                        default="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin",
                        help='cuda home directory')
    parser.add_argument('--CUDA_VER', type=str, default="12.6", help='cuda version')
    parser.add_argument('--GPU_TYPE', type=str, default="RTX 2080 Ti", help='gpu type')
    parser.add_argument('--GPU_ARCH', type=str, default="7.5", help='gpu arch')
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument('--keep_temp', choices=[True, False], default=True, help='keep_temp')
    args = parser.parse_args()
    return args

def main(args):
    llm = HttpsApi(
        host='hk-api.gptbest.vip', key='sk-le1LLTBIQGMfP47XCb924e88919c456aB21eB5Af20E05632',
        model='gpt-4o-2024-08-06', timeout=200
    )

    task = KernelEvaluation(args)

    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir=os.path.join(args.res_path, "logs"), log_style='complex'),
        evaluation=task,
        max_sample_nums=100,
        max_generations=10,
        pop_size=20,
        num_samplers=4,
        num_evaluators=4,
        code_type="Kernel"
    )

    method.run()


if __name__ == '__main__':
    args = parse_args()
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    args.res_path = os.path.join(RES_PATH, time_stamp)
    os.makedirs(args.res_path, exist_ok=True)

    with open(os.path.join(DATA_PATH, 'func.py'), 'r') as f:
        func_code = f.read()

    with open(os.path.join(DATA_PATH, 'test_cuda_code.cu'), 'r') as f:
        cuda_code = f.read()

    args.func_code = func_code
    args.cuda_code = cuda_code
    main(args)