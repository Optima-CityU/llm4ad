# HSEvo: Diversity-Driven Harmony Search + Genetic Algorithm

**HSEvo** (*Harmony Search Evolution*) is an LLM-based evolutionary program search method that combines genetic operators with a **harmony search (HS)** operator for automatic heuristic design. It is integrated into [LLM4AD](https://github.com/Optima-CityU/llm4ad) as `llm4ad.method.hsevo`.

**Paper:** [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://doi.org/10.1609/aaai.v39i25.34898) (AAAI 2025)

**Upstream code:** [datphamvn/HSEvo](https://github.com/datphamvn/hsevo)

---

## Overview

Each HSEvo generation runs:

1. **Selection** — random parent pairs with different objective values
2. **Reflection** — flash + comprehensive reflection prompts on the population
3. **Crossover** — LLM combines two parents into a new heuristic
4. **Mutation** — LLM mutates the elitist using reflection hints
5. **Harmony search** — pick one untuned individual, ask the LLM to expose tunable numeric parameters, then search parameter space with classical HS (`hmcr`, `par`, `bandwidth`)

The HS step is the distinctive operator: the LLM rewrites hardcoded thresholds/weights as function defaults, defines `parameter_ranges`, and HSEvo evaluates multiple parameter settings on the **same** heuristic structure.

Successful HS runs log `[HS-CHECK]` with `distinct_init_objs > 1`, confirming different parameter vectors produced different scores.

---

## Quick Start

Configure your LLM API, then run any task script:

```bash
uv run python example/tasks/online_bin_packing/run_hsevo.py
```

Minimal Python example:

```python
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.hsevo import HSEvo, HSEvoProfiler

llm = HttpsApi(host='xxx', key='sk-xxx', model='xxx', timeout=60)
task = OBPEvaluation()

method = HSEvo(
    llm=llm,
    profiler=HSEvoProfiler(log_dir='logs/hsevo', log_style='simple'),
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
)

method.run()
```

A copy of this OBP script lives at [`run_hsevo_obp.py`](./run_hsevo_obp.py).

---

## Hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_sample_nums` | `450` | Stop after this many function evaluations |
| `pop_size` | `10` | Parents used for crossover each generation |
| `init_pop_size` | `30` | Initial population size (rotating scientist personas) |
| `mutation_rate` | `0.5` | Fraction of `pop_size` mutated from the elitist |
| `hm_size` | `5` | Harmony memory size (initial random parameter vectors) |
| `hmcr` | `0.7` | Harmony memory considering rate |
| `par` | `0.5` | Pitch adjustment rate |
| `bandwidth` | `0.2` | Pitch adjustment bandwidth (fraction of each param range) |
| `max_iter` | `5` | HS improvisation iterations after memory init |
| `num_samplers` | `4` | Parallel LLM sampling threads |
| `num_evaluators` | `4` | Parallel evaluation workers |

Defaults are also listed in [`llm4ad/method/hsevo/paras.yaml`](../../../llm4ad/method/hsevo/paras.yaml).

---

## Task entry scripts

Every task under [`example/tasks/`](../../tasks/) that has `run_eoh.py` also provides `run_hsevo.py`:

| Task | Script |
|------|--------|
| Online bin packing | [`example/tasks/online_bin_packing/run_hsevo.py`](../../tasks/online_bin_packing/run_hsevo.py) |
| TSP constructive | [`example/tasks/tsp_construct/run_hsevo.py`](../../tasks/tsp_construct/run_hsevo.py) |
| CVRP constructive | [`example/tasks/cvrp_construct/run_hsevo.py`](../../tasks/cvrp_construct/run_hsevo.py) |
| QAP | [`example/tasks/qap/run_hsevo.py`](../../tasks/qap/run_hsevo.py) |
| FSSP / JSSP | [`example/tasks/fssp_construct/run_hsevo.py`](../../tasks/fssp_construct/run_hsevo.py) |
| Orienteering | [`example/tasks/orienteering_construct/run_hsevo.py`](../../tasks/orienteering_construct/run_hsevo.py) |
| VRPTW | [`example/tasks/vrptw_construct/run_hsevo.py`](../../tasks/vrptw_construct/run_hsevo.py) |
| Pymoo MOEA/D | [`example/tasks/pymoo_moead/run_hsevo.py`](../../tasks/pymoo_moead/run_hsevo.py) |
| Car racing control | [`example/tasks/control_carracing/run_hsevo.py`](../../tasks/control_carracing/run_hsevo.py) |
| Moon lander control | [`example/tasks/control_moonlander/run_hsevo.py`](../../tasks/control_moonlander/run_hsevo.py) |
| Circle packing | [`example/tasks/circle_packing/EoH_settings&logs/run_hsevo.py`](../../tasks/circle_packing/EoH_settings&logs/run_hsevo.py) |

---

## Logging

`HSEvoProfiler` writes:

- `run_log.txt` — generation progress, `[HS-CHECK]`, `harmony_search: OK/FAILED`
- `samples/` — evaluated programs tagged by `operator` (`init`, `crossover`, `mutation`, `harmony_search`, …)
- `population/` — per-generation population checkpoints

Profiler variants: `HSEvoTensorboardProfiler`, `HSEvoWandbProfiler`.

---

## Citation

```bibtex
@inproceedings{dat2025hsevo,
  title={Hsevo: Elevating automatic heuristic design with diversity-driven harmony search and genetic algorithm using llms},
  author={Dat, Pham Vu Tuan and Doan, Long and Binh, Huynh Thi Thanh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={25},
  pages={26931--26938},
  year={2025}
}
```
