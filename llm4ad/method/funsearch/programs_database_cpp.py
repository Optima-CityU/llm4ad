# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy

from . import config as config_lib
from ...base import ModifyCode, Function, Program, TextFunctionProgramConverter
from ...base.opt_kernels import  KERTextFunctionProgramConverter


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`.
    """
    try:
        if not np.all(np.isfinite(logits)):
            non_finites = set(logits[~np.isfinite(logits)])
            raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
        if not np.issubdtype(logits.dtype, np.floating):
            logits = np.array(logits, dtype=np.float32)

        result = scipy.special.softmax(logits / temperature, axis=-1)
        # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
        index = np.argmax(result)
        result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
        return result
    except TypeError as type_err:
        print(logits)
        raise type_err


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated funsearch_impl
         into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ProgramsDatabaseConfig,
            template: Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: Program = template
        self._function_to_evolve: str = function_to_evolve
        # Initialize empty islands.
        self._islands: list[Island_cpp] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island_cpp(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period)
            )
        self._best_score_per_island: list[float] = ([-float('inf')] * config.num_islands)
        self._best_program_per_island: list[Function | None] = ([None] * config.num_islands)
        self._best_scores_per_test_per_island: list[Any | None] = ([None] * config.num_islands)
        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    @property
    def islands(self):
        return self._islands

    def _register_function_in_island(
            self,
            function: Function,
            island_id: int,
            score: int | float,
    ) -> None:
        """Registers `function` in the specified island."""
        self._islands[island_id].register_function(function, score)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = function
            self._best_scores_per_test_per_island[island_id] = score
            self._best_score_per_island[island_id] = score

    def register_function(
            self,
            function: Function,
            island_id: int | None,
            score: Any,
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous funsearch_impl we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_function_in_island(function, island_id, score)
        else:
            self._register_function_in_island(function, island_id, score)

        # Check whether it is time to reset an island.
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island_cpp(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_function_in_island(founder, island_id, founder_scores)


class Island_cpp:
    """A sub-population of the programs database."""

    def __init__(
            self,
            template: Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (cluster_sampling_temperature_period)
        self._clusters: dict[Any, Cluster] = {}
        self._num_programs: int = 0

    @property
    def clusters(self):
        return self._clusters

    def get_num_programs(self) -> int:
        """Implemented by RZ, help us to profile.
        """
        return self._num_programs

    def register_function(
            self,
            function: Function,
            score: int | float,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = score
        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, function)
        else:
            self._clusters[signature].register_program(function)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        # ------------------------------------------------------------------------------
        # Normalized the score
        # ------------------------------------------------------------------------------
        max_abs_score = float(np.abs(cluster_scores).max())
        if max_abs_score > 1:
            cluster_scores = cluster_scores.astype(float) / max_abs_score
        # ------------------------------------------------------------------------------

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(self, implementations: Sequence[Function]) -> str:
        assert len(implementations) in [1, 2], 'Rui: We assume the prompt parameter `k` equals to 1 or 2.'
        if len(implementations) == 1:
            prompt = f"""
You are a Machine Learning Engineer trying to reduce the runtime of a {implementations[0].name} kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a RTX 4090 GPU with CUDA 12.4.

Answer using the following schema:

```cpp
[Your {implementations[0].name}` kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the CUDA kernel code example you need to optimize:
```cpp
{str(implementations[0])}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        else:
            prompt = f"""
You are a Machine Learning Engineer trying to reduce the runtime of a {implementations[0].name} kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a RTX 4090 GPU with CUDA 12.4.

Answer using the following schema:

```cpp
[Your {implementations[0].name}` kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is a CUDA kernel code example:
```cpp
{str(implementations[0])}
```

A better version of the CUDA kernel code example is as follows:
```cpp
{str(implementations[1])}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        return prompt


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: Function):
        self._score = score
        self._programs: list[Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    @property
    def programs(self):
        return self._programs

    def register_program(self, program: Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)  # noqa
