from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *

class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        # TODO: only to 2 objectives
        w1 = np.linspace(0, 1, self._pop_size)
        self._weight_vectors = np.array([w1, 1 - w1])
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = [float('-inf'), float('-inf')]
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = [float('-inf'), float('-inf')]
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                crt_pop_size = len(pop)
                selected_idx_list = []
                for i in range(self._pop_size):
                    best_sub_score = float('-inf')
                    best_sub_idx = None
                    for j in range(crt_pop_size):
                        sub_score = -np.max(-self._weight_vectors[:, i] * np.array(pop[j].score)) # TCH
                        if best_sub_score < sub_score and np.isfinite(sub_score):
                            best_sub_score = sub_score
                            best_sub_idx = j
                    selected_idx_list.append(best_sub_idx)
                self._population = [pop[i] for i in selected_idx_list]
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self, pref: np.array) -> Function:
        funcs = [f for f in self._population if not np.isinf(np.array(f.score)).any()]

        crt_pop_size = len(funcs)
        sub_score_list = []
        sub_idx_list = []
        for j in range(crt_pop_size):
            sub_score = -np.max(-pref * np.array(funcs[j].score)) # TCH
            if np.isfinite(sub_score):
                sub_score_list.append(sub_score)
                sub_idx_list.append(j)
        sub_score_list = np.array(sub_score_list)
        sub_idx_list = np.array(sub_idx_list)
        sorted_idx = np.argsort(-sub_score_list) # minus for descending
        sub_idx_list = sub_idx_list[sorted_idx]
        func = [funcs[i] for i in sub_idx_list]
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)
        return np.random.choice(func, p=p)
