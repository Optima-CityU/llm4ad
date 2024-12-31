from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from threading import Lock

from .population import RegEvoPopulation
from ...tools.profiler import ProfilerBase
from ...tools.profiler import TensorboardProfiler
from ...tools.profiler import WandBProfiler


class RegEvoProfiler(ProfilerBase, ABC):
    @abstractmethod
    def register_regevo_pop(self, pop: RegEvoPopulation):
        pass


class RegEvoTensorboardProfiler(TensorboardProfiler, RegEvoProfiler):
    _pop_order = 0

    def __init__(
            self,
            log_dir: str | None = None,
            *,
            initial_num_samples=0,
            pop_register_interval: int = 100,
            log_style='complex'
    ):
        """
        Args:
            log_dir: log file path
            pop_register_interval: log the ProgramDB after getting N samples each time
        """
        RegEvoProfiler.__init__(self)
        TensorboardProfiler.__init__(self, log_dir, initial_num_samples=initial_num_samples, log_style=log_style)
        if log_dir:
            self._pop_path = os.path.join(log_dir, 'population')
            os.makedirs(self._pop_path, exist_ok=True)
        self._intv = pop_register_interval
        self._pop_lock = Lock()

    def register_regevo_pop(self, pop: RegEvoPopulation):
        """Save population to a file."""
        try:
            if (self.__class__._num_samples == 0 or
                    self.__class__._num_samples % self._intv != 0):
                return
            self._pop_lock.acquire()
            self.__class__._pop_order += 1

            clus_scores = {}
            for k, v in pop.cluster_scores:
                funcs = [str(f) for f in v.funcs]
                clus_scores[k] = funcs

            path = os.path.join(self._pop_path, f'pop_{self.__class__._pop_order}.json')
            with open(path, 'w') as f:
                json.dump(clus_scores, f)
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()


class RegEvoWandbProfiler(WandBProfiler, RegEvoProfiler):
    _pop_order = 0

    def __init__(
            self,
            wandb_project_name: str,
            log_dir: str | None = None,
            *,
            initial_num_samples=0,
            pop_register_interval: int = 100,
            log_style='complex',
            **kwargs
    ):
        """
        Args:
            log_dir: log file path
            pop_register_interval: log the ProgramDB after getting N samples each time
        """
        RegEvoProfiler.__init__(self)
        WandBProfiler.__init__(self, wandb_project_name, log_dir, initial_num_samples=initial_num_samples, log_style=log_style, **kwargs)
        if log_dir:
            self._pop_path = os.path.join(log_dir, 'population')
            os.makedirs(self._pop_path, exist_ok=True)
        self._intv = pop_register_interval
        self._pop_lock = Lock()

    def register_regevo_pop(self, pop: RegEvoPopulation):
        """Save population to a file."""
        try:
            if (self.__class__._num_samples == 0 or
                    self.__class__._num_samples % self._intv != 0):
                return
            self._pop_lock.acquire()
            self.__class__._pop_order += 1

            clus_scores = {}
            for k, v in pop.cluster_scores:
                funcs = [str(f) for f in v.funcs]
                clus_scores[k] = funcs

            path = os.path.join(self._pop_path, f'pop_{self.__class__._pop_order}.json')
            with open(path, 'w') as f:
                json.dump(clus_scores, f)
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()
