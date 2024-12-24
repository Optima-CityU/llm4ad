from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.randomized_argsort import randomized_argsort

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
                # modified from pymoo.algorithms.moo.nsga2
                # get the objective space values and objects
                objs = [ind.score for ind in pop]
                F = -np.array(objs)

                # the final indices of surviving individuals
                survivors = []

                # do the non-dominated sorting until splitting front
                fronts = NonDominatedSorting().do(F, n_stop_if_ranked=self._pop_size)

                for k, front in enumerate(fronts):

                    I = np.arange(len(front))

                    # current front sorted by crowding distance if splitting
                    if len(survivors) + len(I) > self._pop_size:

                        # Define how many will be removed
                        n_remove = len(survivors) + len(front) - self._pop_size

                        # re-calculate the crowding distance of the front
                        crowding_of_front = \
                            get_crowding_function("cd").do(
                                F[front, :],
                                n_remove=n_remove
                            )

                        I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                        I = I[:-n_remove]

                    # otherwise take the whole front unsorted
                    else:
                        # calculate the crowding distance of the front
                        crowding_of_front = \
                            get_crowding_function("cd").do(
                                F[front, :],
                                n_remove=0
                            )

                    # save rank and crowding in the individual class
                    for j, i in enumerate(front):
                        pop[i].rank = k
                        pop[i].crowding = crowding_of_front[j]

                    # extend the survivors by all or selected individuals
                    survivors.extend(front[I])

                self._population = [pop[i] for i in survivors]
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

    def selection(self) -> Function:
        # funcs = [f for f in self._population if not math.isinf(f.score)]
        funcs = [f for f in self._population if not np.isinf(np.array(f.score)).any()]
        if len(funcs) > 1:
            parents = np.random.choice(funcs, size=2, replace=False)
            if parents[0].rank < parents[1].rank:
                return parents[0]
            elif parents[0].rank > parents[1].rank:
                return parents[1]
            elif parents[0].crowding > parents[1].crowding:
                return parents[0]
            else:
                return parents[1]
        else:
            return funcs[0]
