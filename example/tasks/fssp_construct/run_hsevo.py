from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from llm4ad.task.optimization.jssp_construct import JSSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.hsevo import HSEvo, HSEvoProfiler


def main():
    llm = HttpsApi(
        host="xxx",  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
        key="sk-xxx",  # your key, e.g., 'sk-abcdefghijklmn'
        model="xxx",  # your llm, e.g., 'gpt-3.5-turbo'
        timeout=120,
    )

    task = JSSPEvaluation()

    method = HSEvo(
        llm=llm,
        profiler=HSEvoProfiler(log_dir="logs/hsevo", log_style="simple"),
        evaluation=task,
        max_sample_nums=100,
        pop_size=4,
        init_pop_size=10,
        mutation_rate=0.5,
        hm_size=5,
        hmcr=0.7,
        par=0.5,
        bandwidth=0.2,
        max_iter=5,
        num_samplers=4,
        num_evaluators=4,
        debug_mode=False,
    )

    method.run()


if __name__ == "__main__":
    main()
