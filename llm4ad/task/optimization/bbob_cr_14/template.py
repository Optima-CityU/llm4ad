template_program = '''
import numpy as np
from typing import Tuple, List
def crossover(self, x_mu=None, x=None, r=None):
    """
    Crossover the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - h: int, Length of historical memory.
            - p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
        x_mu: The mutated population of individuals, shape=(self.n_individuals, self.ndim_problem).
        x: The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        r: The indices of the selected individuals used for mutation and crossover, shape=(self.n_individuals,).

    Returns:
        x_cr: The crossover population of individuals, shape=(self.n_individuals, self.ndim_problem).
        p_cr: The crossover probabilities for each individual, shape=(self.n_individuals,).
    """
    
    x_cr = np.copy(x)
    p_cr = np.empty((self.n_individuals,))  # crossover probabilities
    for k in range(self.n_individuals):
        p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
        p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)
        i_rand = self.rng_optimization.integers(self.ndim_problem)
        for i in range(self.ndim_problem):
            if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                x_cr[k, i] = x_mu[k, i]
    return x_cr, p_cr
'''

task_description = "Implement a crossover operator for black-box optimization."


class TEMP:
    def __init__(self):
        self.test = None
        self.test1 = 'c'

    def t(self):
        print("a")

if __name__ == "__main__":
    a = TEMP()
    a.__setattr__('mutate', template_program)
    tt = '''
import numpy as np
def t(self):
    print(self.test1)
    '''
    all_globals_namespace = {}
    # execute the program, map func/var/class to global namespace
    exec(tt, all_globals_namespace)
    # get the pointer of 'function_to_run'
    program_callable = all_globals_namespace['t']
    from types import MethodType
    a.t = MethodType(program_callable, a)
    print(a)
