from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Dict

from .population import Population
from ...base import Function
from ...tools.profiler import TensorboardProfiler, ProfilerBase, WandBProfiler


class EoHProfiler(ProfilerBase, ABC):
    @abstractmethod
    def register_population(self, population: Population):
        pass


class EoHTensorboardProfiler(TensorboardProfiler, EoHProfiler):
    _cur_gen = 0

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex'):
        EoHProfiler.__init__(self)
        TensorboardProfiler.__init__(self, log_dir, initial_num_samples=initial_num_samples, log_style=log_style)
        self._pop_lock = Lock()
        if log_dir:
            self._ckpt_dir = os.path.join(log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self.__class__._num_samples == 0 or
                    pop.generation == self.__class__._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_json = {
                    'function': str(f),
                    'score': f.score,
                    'algorithm': f.algorithm
                }
                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file)
            self.__class__._cur_gen += 1
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()


class EoHWandbProfiler(WandBProfiler, EoHProfiler):
    _cur_gen = 0

    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 **kwargs):
        EoHProfiler.__init__(self)
        WandBProfiler.__init__(self,
                               wandb_project_name,
                               log_dir,
                               initial_num_samples=initial_num_samples,
                               log_style=log_style, **kwargs)
        self._pop_lock = Lock()
        if log_dir:
            self._ckpt_dir = os.path.join(log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self.__class__._num_samples == 0 or
                    pop.generation == self.__class__._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_json = {
                    'function': str(f),
                    'score': f.score,
                    'algorithm': f.algorithm
                }
                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file)
            self.__class__._cur_gen += 1
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()
