import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import ProfilerBase


def main():
    llm = HttpsApi(host="xxx",  # your host endpoint, e.g., api.openai.com, api.deepseek.com
                   key="xxx",  # your key, e.g., sk-xxxxxxxxxx
                   model="gpt-3.5-turbo",  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=20)
    
    task = OBPEvaluation()

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs/eoh', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=20000,
                 max_generations=10,
                 pop_size=4,
                 num_samplers=1,
                 num_evaluators=1, 
                 debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
