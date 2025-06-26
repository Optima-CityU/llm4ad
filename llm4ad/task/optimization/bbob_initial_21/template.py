template_program = '''
import numpy as np
from typing import Tuple, List
def initialize(self, args=None):
    """
    Crossover the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - initial_lower_boundary: int, Lower boundary for the initial population.
            - initial_upper_boundary: int, Upper boundary for the initial population.
            - _check_terminations(): method, 
                    Method to check termination conditions, 
                    Return True if reach the terminations else False.
            - h: int, Length of historical memory.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - _evaluate_fitness(individual: np.ndarray, args): method, 
                    Method to evaluate the fitness of an individual, input individual shape=(self.ndim_problem,).
                    Return fitness value of the individual.
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).

    Returns:
        x: np.ndarray, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.ndarray, The current fitness values of the population, shape=(self.n_individuals,).
        a: np.ndarray, The archive of inferior solutions, shape=(0, self.ndim_problem).
    """
    x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                        size=(self.n_individuals, self.ndim_problem))  # population
    y = np.empty((self.n_individuals,))  # fitness
    for i in range(self.n_individuals):
        if self._check_terminations():
            break
        y[i] = self._evaluate_fitness(x[i], args)
    a = np.empty((0, self.ndim_problem))  # set of archived inferior solutions
    return x, y, a
'''

task_description = "Implement an effective initialization operator for the black-box optimization."


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
