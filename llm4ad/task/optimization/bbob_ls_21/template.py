template_program = '''
import numpy as np
from typing import Tuple, List
def local_search(self, x_cr=None, args=None):
    """
    Proceed local search for the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - h: int, Length of historical memory.
            - p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_lower_boundary: int, Lower boundary for the initial population.
            - initial_upper_boundary: int, Upper boundary for the initial population.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
            - _check_terminations(): method, 
                    Method to check termination conditions, 
                    Return True if reach the terminations else False.
            - _evaluate_fitness(individual: np.ndarray, args): method, 
                    Method to evaluate the fitness of an individual, input individual shape=(self.ndim_problem,).
                    Return fitness value of the individual.
        x_cr: The crossover population of individuals, shape=(self.n_individuals, self.ndim_problem).
        
    Returns:
        x_ls: The population of individuals after local search, shape=(self.n_individuals, self.ndim_problem).
        y_ls: The fitness values of the population after local search, shape=(self.n_individuals,).
    """
    y_ls = np.empty((self.n_individuals,))
    x_ls = x_cr
    for k in range(self.n_individuals):
        if self._check_terminations():
            break
        y_ls[k] = self._evaluate_fitness(x_ls[k], args)
    return x_ls, y_ls
'''

task_description = "Implement a local search operator for black-box optimization."


# def mutate(self, x: np.ndarray, y: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Mutate the population.
#
#     Args:
#         self: The instance of the class containing the mutation parameters and methods.
#             - n_individuals: int, Number of individuals in the population.
#             - ndim_problem: int, Dimension of the problem.
#             - h: int, Length of historical memory.
#             - p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
#             - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
#             - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
#         x: The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
#         y: The current fitness values of the population, shape=(self.n_individuals,).
#         a: The archive of inferior solutions, shape=(self.n_individuals, self.ndim_problem).
#
#     Returns:
#         x_mu: The mutated population of individuals, shape=(self.n_individuals, self.ndim_problem).
#         f_mu: The mutated mutation factors for each individual, shape=(self.n_individuals,).
#         r: The indices of the selected individuals used for mutation and crossover, shape=(self.n_individuals,).
#     """
#     x_mu = np.empty((self.n_individuals, self.ndim_problem))  # mutated population
#     f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
#     x_un = np.vstack((np.copy(x), a))  # union of population x and archive a
#     r = self.rng_optimization.choice(self.h, (self.n_individuals,))
#     order = np.argsort(y)[:]
#     p = (0.2 - self.p_min) * self.rng_optimization.random((self.n_individuals,)) + self.p_min
#     idx = [order[self.rng_optimization.choice(int(i))] for i in np.ceil(p * self.n_individuals)]
#     for k in range(self.n_individuals):
#         f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
#         while f_mu[k] <= 0.0:
#             f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
#         if f_mu[k] > 1.0:
#             f_mu[k] = 1.0
#         r1 = self.rng_optimization.choice([i for i in range(self.n_individuals) if i != k])
#         r2 = self.rng_optimization.choice([i for i in range(len(x_un)) if i != k and i != r1])
#         x_mu[k] = x[k] + f_mu[k] * (x[idx[k]] - x[k]) + f_mu[k] * (x[r1] - x_un[r2])
#
#     return x_mu, f_mu, r


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
