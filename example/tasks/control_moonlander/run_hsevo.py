from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from llm4ad.task.machine_learning.moon_lander import (
    MoonLanderEvaluation,
    moon_lander_feature,
)
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.hsevo import HSEvo, HSEvoProfiler


def main():
    llm = HttpsApi(
        host="xxx",  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
        key="sk-xxx",  # your key, e.g., 'sk-abcdefghijklmn'
        model="xxx",  # your llm, e.g., 'gpt-3.5-turbo'
        timeout=120,
    )

    seeds = [
        6,
        9,
        17,
        29,
        57,
        44,
        18,
        69,
        26,
        68,
        65,
        23,
        51,
        93,
        16,
        87,
        92,
        90,
        22,
        73,
        60,
        10,
        19,
        97,
        11,
        14,
        99,
        98,
        8,
        28,
        43,
        56,
        89,
        15,
        74,
    ]
    instance_set = {idx: seed for idx, seed in enumerate(seeds)}
    using_seeds = list(range(100, 150))
    ins_to_be_solve_set = {idx: seed for idx, seed in enumerate(using_seeds)}

    task = MoonLanderEvaluation(
        whocall="eoh",
        instance_set=instance_set,
        run_mode="Training",
        ins_to_be_solve_set=ins_to_be_solve_set,
        feature_pipeline=moon_lander_feature,
        objective_value=230,
    )

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
