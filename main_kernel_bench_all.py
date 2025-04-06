import os
import time
import natsort
import argparse



from llm4ad.task.engineering.opt_kernels import KernelEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

# from llm4ad.task.engineering.opt_kernels.preprocess.pipeline_func import convert_to_functional_code, translate_into_CUDA_kernel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on KernelBench')
    # My local computer
    # parser.add_argument('--CUDA_HOME', type=str, default="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin", help='cuda home directory')
    # parser.add_argument('--CUDA_VER', type=str, default="12.6", help='cuda version')
    # parser.add_argument('--GPU_TYPE', type=str, default="RTX 4060 Ti", help='gpu type')
    # parser.add_argument('--GPU_ARCH', type=str, default="8.9", help='gpu arch')
    # computer of CityU
    # parser.add_argument('--CUDA_HOME', type=str, default="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin", help='cuda home directory')
    # parser.add_argument('--CUDA_VER', type=str, default="12.6", help='cuda version')
    # parser.add_argument('--GPU_TYPE', type=str, default="RTX 2080 Ti", help='gpu type')
    # parser.add_argument('--GPU_ARCH', type=str, default="7.5", help='gpu arch')

    # Cloud Computer 4090
    parser.add_argument('--CUDA_HOME', type=str, default="/usr/local/cuda", help='cuda home directory')
    parser.add_argument('--CUDA_VER', type=str, default="12.4", help='cuda version')
    parser.add_argument('--GPU_TYPE', type=str, default="RTX 4090", help='gpu type')
    parser.add_argument('--GPU_ARCH', type=str, default="8.9", help='gpu arch')


    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument('--keep_temp', choices=[True, False], default=True, help='keep_temp')
    args = parser.parse_args()
    return args

def main(args):
    os.environ["CUDA_HOME"] = args.CUDA_HOME
    os.environ['TORCH_CUDA_ARCH_LIST'] = args.GPU_ARCH
    # get cpu count
    cpu_count = os.cpu_count()
    os.environ['MAX_JOBS'] = f"{cpu_count}"

    # deepseek_config_dict = {
    #     "host": "api.deepseek.com", "key": "sk-60c9ae55582545dba2a72c3a4b498e82", "timeout": 120
    # }
    # ds_r1_instance = HttpsApi(model="o3-mini", **deepseek_config_dict)
    # o3_llm = HttpsApi(
    #     host='hk-api.gptbest.vip', key='sk-le1LLTBIQGMfP47XCb924e88919c456aB21eB5Af20E05632',
    #     model='o3-mini', timeout=200
    # )
    #
    # llm = HttpsApi(
    #     host='hk-api.gptbest.vip', key='sk-le1LLTBIQGMfP47XCb924e88919c456aB21eB5Af20E05632',
    #     model='gpt-4o-2024-08-06', timeout=200
    # )

    ds_v3 = HttpsApi(
        host='api.deepseek.com', key='sk-60c9ae55582545dba2a72c3a4b498e82',
        model='deepseek-chat', timeout=300
    )

    task = KernelEvaluation(args)

    method = EoH(
        llm=ds_v3,
        profiler=EoHProfiler(log_dir=os.path.join(args.res_path, "logs"), log_style='complex'),
        evaluation=task,
        max_sample_nums=45,
        max_generations=9,
        pop_size=5,
        num_samplers=4,
        num_evaluators=1,
        code_type="Kernel"
    )

    method.run()

# use the absolute path to avoid the path error
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(ABS_PATH, 'Results')
# DATA_PATH = os.path.join(ABS_PATH, 'init_dataset', 'level1', '1_Square_matrix_multiplication_', "CudaCodeVerify")
DATA_PATH = os.path.join(ABS_PATH, 'init_dataset', 'level1')

if __name__ == '__main__':
    args = parse_args()
    # time_stamp = time.strftime("%Y%m%d-%H%M%S")
    time_stamp = "20250405-161548"

    operation_list = os.listdir(DATA_PATH)
    operation_list = natsort.natsorted(operation_list)
    for each_operation in operation_list:
        args.res_path = os.path.join(RES_PATH, time_stamp, each_operation)
        if os.path.exists(args.res_path):
            continue
        os.makedirs(args.res_path, exist_ok=True)
        func_file_path = os.path.join(DATA_PATH, each_operation, "CudaCodeVerify", 'func.py')
        cuda_file_path = os.path.join(DATA_PATH, each_operation, "CudaCodeVerify", 'test_cuda_code.cu')
        if not os.path.exists(func_file_path) or not os.path.exists(cuda_file_path):
            continue

        with open(func_file_path, 'r') as f:
            func_code = f.read()

        with open(cuda_file_path, 'r') as f:
            cuda_code = f.read()

        args.code_operation = each_operation
        args.func_code = func_code
        args.cuda_code = cuda_code
        main(args)