import numpy as np  # engine for numerical computing

# abstract class of all Differential Evolution (DE) classes
from de import DE


class CDE(DE):
    """Classic Differential Evolution (CDE).

    .. note:: Typically, `DE/rand/1/bin` is seen as the **classic/basic** version of `DE`.
       `CDE` often optimizes on relatively low-dimensional (e.g., << 1000) search spaces.
       Its two creators (Kenneth Price&Rainer Storn) won the 2017 Evolutionary
       Computation Pioneer Award from IEEE-CIS.

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective/cost function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`,
                  default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`,
                  default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed
                  to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size
                  (`int`, default: `100`),
                * 'f'             - mutation factor (`float`, default: `0.5`),
                * 'cr'            - crossover probability (`float`, default: `0.9`).

    Examples
    --------
    Use the optimizer `CDE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:


       CDE: 5000, 2.0242437417701847e-07

    For its correctness checking of Python coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3fc826yt>`_ for more details.

    Attributes
    ----------
    cr            : `float`
                    crossover probability.
    f             : `float`
                    mutation factor.
    n_individuals : `int`
                    number of offspring, aka offspring population size.

    References
    ----------
    Price, K.V., 2013.
    `Differential evolution.
    <https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8>`_
    In Handbook of Optimization (pp. 187-214). Springer.

    Price, K.V., Storn, R.M. and Lampinen, J.A., 2005.
    `Differential evolution: A practical approach to global optimization.
    <https://link.springer.com/book/10.1007/3-540-31306-0>`_
    Springer Science & Business Media.

    Storn, R.M. and Price, K.V. 1997.
    `Differential evolution – A simple and efficient heuristic for global
    optimization over continuous spaces.
    <https://link.springer.com/article/10.1023/A:1008202821328>`_
    Journal of Global Optimization, 11(4), pp.341–359.
    (Kenneth Price&Rainer Storn won the **2017** `Evolutionary
    Computation Pioneer Award from IEEE CIS
    <https://cis.ieee.org/awards/past-recipients>`_.)
    """
    def __init__(self, problem, options, rand_seed):
        DE.__init__(self, problem, options)
        self.f = options.get('f', 0.5)  # mutation factor
        self.cr = options.get('cr', 0.9)  # crossover probability

        # set seed
        self.rng_initialization = np.random.default_rng(rand_seed)
        self.rng_optimization = np.random.default_rng(rand_seed)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        v = np.empty((self.n_individuals, self.ndim_problem))  # for mutation
        return x, y, v

    def mutate(self, x=None, v=None):
        for i in range(self.n_individuals):
            r = self.rng_optimization.permutation(self.n_individuals)[:4]
            r = r[r != i][:3]  # a simple yet effective trick
            v[i] = x[r[0]] + self.f*(x[r[1]] - x[r[2]])
        return v

    def crossover(self, v=None, x=None):
        """Binomial crossover (uniform discrete crossover)."""
        for i in range(self.n_individuals):
            j_r = self.rng_optimization.integers(self.ndim_problem)
            # to avoid loop (a simple yet effective trick)
            tmp = v[i, j_r]
            co = self.rng_optimization.random(self.ndim_problem) > self.cr
            v[i, co] = x[i, co]
            v[i, j_r] = tmp
        return v

    def select(self, v=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(v[i], args)
            if yy < y[i]:
                x[i], y[i] = v[i], yy
        return x, y

    def iterate(self, x=None, y=None, v=None, args=None):
        v = self.mutate(x, v)
        v = self.crossover(v, x)
        x, y = self.select(v, x, y, args)
        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, v = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, v, args)
        return self._collect(fitness, y)