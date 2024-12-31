from __future__ import annotations

import random
from threading import Lock
from typing import Dict, List

import numpy as np
import scipy

from .config import RegEvoPopulationConfig
from ...base import Function


class RegEvoPopulation:
    def __init__(self, pop_size, config=RegEvoPopulationConfig()):
        self._pop_size = pop_size
        self._cluster_scores: Dict[int, Cluster] = {}
        self._cluster_sampling_temperature_init = config.cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = config.cluster_sampling_temperature_period
        self._num_funcs = 0
        self._register_function_lock = Lock()
        self._selection_lock = Lock()

    @property
    def num_funcs(self):
        return self._num_funcs

    @property
    def cluster_scores(self):
        return self._cluster_scores

    @property
    def scores(self):
        return list(self._cluster_scores.keys())

    @property
    def cur_pop_size(self):
        return sum([len(v) for k, v in self._cluster_scores.items()])

    @property
    def max_age(self):
        return max([v.max_age for k, v in self._cluster_scores.items()])

    def __len__(self):
        return len(list(self._cluster_scores.keys()))
        # return sum(len(v) for k, v in self._cluster_scores.items())

    def register_function(self, func: Function):
        """Register a function to the solution set.
        """
        try:
            self._register_function_lock.acquire()
            score = func.score
            assert score is not None
            assert isinstance(score, int) or isinstance(score, float)
            self._num_funcs += 1
            score = np.round(score, 5)
            func.age = 0
            if score in self._cluster_scores:
                # register the func to the cluster
                self._cluster_scores[score].register_function(func)
            else:
                # create a new cluster
                clus = Cluster(score=score)
                clus.register_function(func)
                self._cluster_scores[score] = clus
            self._update_pop()
        finally:
            self._register_function_lock.release()

    def _update_pop(self):
        # add age for each individual in the population
        for s in self.scores:
            for f in self._cluster_scores[s].funcs:
                f.age += 1
        # discard nothing if the population is not full
        if self.cur_pop_size <= self._pop_size:
            return
        # remove the old individual in the entire population
        cluster_scores = [k for k, v in self._cluster_scores.items()
                          if v.max_age == self.max_age]
        # random select a cluster to kill
        killed_id = random.choice(cluster_scores)
        self._cluster_scores[killed_id].kill_oldest_indiv()
        # if the cluster has no individuals, remove it from the population
        if len(self._cluster_scores[killed_id]) == 0:
            self._cluster_scores.pop(killed_id)

    # def get_top_k_funcs(self, k) -> List[Function]:
    #     """Get k functions with the highest scores.
    #     """
    #     all_funcs = []
    #     for s in self._cluster_scores.keys():
    #         funcs = self._cluster_scores[s]
    #         all_funcs += funcs
    #     all_funcs = sorted(all_funcs, key=lambda f: f.score, reverse=True)
    #     return all_funcs[:k]

    def selection(self, num: int):
        try:
            self._selection_lock.acquire()
            cluster_scores = self._select_clusters(num)
            selected_funcs = []
            for s in cluster_scores:
                f = self._cluster_scores[s].selection()
                selected_funcs.append(f)
            return selected_funcs
        finally:
            self._selection_lock.release()

    def _select_clusters(self, num) -> List[int]:
        """Returns the ids of selected clusters.
        """
        period = self._cluster_sampling_temperature_period
        cluster_scores = np.array(list(self._cluster_scores.keys()))
        # normalize score
        max_abs_score = float(np.abs(cluster_scores).max())
        if max_abs_score > 1:
            normed_cluster_scores = cluster_scores.astype(float) / max_abs_score
        else:
            normed_cluster_scores = cluster_scores
        # calculate temperature
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_funcs % period) / period)
        probabilities = _softmax(normed_cluster_scores, temperature)
        return np.random.choice(a=cluster_scores, size=num, p=probabilities).tolist()


class Cluster:
    """A Cluster records functions with same scores."""

    def __init__(self, score):
        self.score = score
        self.funcs: List[Function] = []

    def register_function(self, func: Function):
        assert hasattr(func, 'age')
        assert func.age is not None
        score = func.score
        assert score is not None
        assert isinstance(score, int) or isinstance(score, float)
        self.funcs.append(func)

    def __getitem__(self, item):
        return self.funcs[item]

    def __len__(self):
        return len(self.funcs)

    def selection(self) -> Function:
        """Samples a program, giving higher probability to shorther programs.
        """
        lengths = [len(str(f).splitlines()) for f in self.funcs]
        normalized_lengths = (np.array(lengths) - min(lengths)) / (
                max(lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self.funcs, p=probabilities)  # noqa

    @property
    def max_age(self):
        return max([f.age for f in self.funcs])

    def kill_oldest_indiv(self):
        funcs = sorted(self.funcs, key=lambda f: f.age)
        self.funcs = funcs[:len(funcs) - 1]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite 'logits'.
    """
    try:
        if not np.all(np.isfinite(logits)):
            non_finites = set(logits[~np.isfinite(logits)])
            raise ValueError(f'"logits" contains non-finite value(s): {non_finites}')
        if not np.issubdtype(logits.dtype, np.floating):
            logits = np.array(logits, dtype=np.float32)
        result = scipy.special.softmax(logits / temperature, axis=-1)
        # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
        index = np.argmax(result)
        result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
        return result
    except TypeError as type_err:
        raise type_err
