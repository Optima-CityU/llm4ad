from __future__ import annotations

import json
import math
import os
from threading import Lock
from typing import List, Dict, Optional

try:
    import wandb
except:
    pass

from ...base import Function
from ...tools.profiler import TensorboardProfiler, ProfilerBase, WandBProfiler


def _serialize_population(population: List[Dict]) -> List[Dict]:
    funcs_json = []
    for ind in population:
        obj = ind.get("obj")
        # JSON cannot represent inf cleanly; store None for invalid individuals.
        if obj is None or (
            isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj))
        ):
            score = None
        else:
            # LLM4AD score (higher is better) = -obj.
            score = -obj
        funcs_json.append(
            {
                "code": ind.get("code"),
                "score": score,
                "obj": None if score is None else obj,
                "exec_success": bool(ind.get("exec_success", False)),
                "tryHS": bool(ind.get("tryHS", False)),
            }
        )
    return funcs_json


class HSEvoProfiler(ProfilerBase):
    def __init__(
        self,
        log_dir: Optional[str] = None,
        *,
        initial_num_samples=0,
        log_style="complex",
        create_random_path=True,
        **kwargs,
    ):
        """HSEvo Profiler.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        super().__init__(
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )
        self._cur_gen = 0
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def register_population(self, population: List[Dict], generation: int):
        """Checkpoint the current HSEvo population (list of individual dicts)."""
        try:
            self._pop_lock.acquire()
            if not self._log_dir:
                return
            if self._num_samples == 0 or generation == self._cur_gen:
                return
            path = os.path.join(self._ckpt_dir, f"pop_{generation}.json")
            with open(path, "w") as json_file:
                json.dump(_serialize_population(population), json_file, indent=4)
            self._cur_gen = generation
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _write_json(
        self, function: Function, program="", *, record_type="history", record_sep=200
    ):
        """Write function data to a JSON file.
        Args:
            function   : The function object containing score and string representation.
            record_type: Type of record, 'history' or 'best'. Defaults to 'history'.
            record_sep : Separator for history records. Defaults to 200.
        """
        assert record_type in ["history", "best"]

        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            "sample_order": sample_order,
            "algorithm": function.algorithm,
            "function": str(function),
            "score": function.score,
            "operator": function.operator,
            "program": program,
        }

        if record_type == "history":
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f"samples_{lower_bound + 1}~{upper_bound}.json"
        else:
            filename = "samples_best.json"

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)


class HSEvoTensorboardProfiler(TensorboardProfiler, HSEvoProfiler):

    def __init__(
        self,
        log_dir: str | None = None,
        *,
        initial_num_samples=0,
        log_style="complex",
        create_random_path=True,
        **kwargs,
    ):
        """Profiler for Tensorboard.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        HSEvoProfiler.__init__(
            self, log_dir=log_dir, create_random_path=create_random_path, **kwargs
        )
        TensorboardProfiler.__init__(
            self,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )

    def finish(self):
        if self._log_dir:
            self._writer.close()

        filename = "end.json"
        path = os.path.join(os.path.join(self._log_dir, "population"), filename)

        with open(path, "w") as json_file:
            json.dump([], json_file, indent=4)


class HSEvoWandbProfiler(WandBProfiler, HSEvoProfiler):
    _cur_gen = 0

    def __init__(
        self,
        wandb_project_name: str,
        log_dir: str | None = None,
        *,
        initial_num_samples=0,
        log_style="complex",
        create_random_path=True,
        **kwargs,
    ):
        """Profiler for Wandb.
        Args:
            wandb_project_name : the name of the wandb project
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        HSEvoProfiler.__init__(
            self, log_dir=log_dir, create_random_path=create_random_path, **kwargs
        )
        WandBProfiler.__init__(
            self,
            wandb_project_name=wandb_project_name,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def finish(self):
        wandb.finish()
        filename = "end.json"
        path = os.path.join(os.path.join(self._log_dir, "population"), filename)

        with open(path, "w") as json_file:
            json.dump([], json_file, indent=4)
