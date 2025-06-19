FunSearch
===============

The `FunSearch` class implements a function search algorithm to optimize a given program using sampling and evaluation.

Usage
-----

To use the `FunSearch` class, initialize it with the required parameters and call the `run` method to start the optimization process.

Constructor
-----------

.. class:: FunSearch

    .. rubric:: Parameters

    - **template_program** (str): The seed program as the initial function of the run. The program should be executable and include necessary imports and definitions.
    - **sampler** (Sampler): An instance of `alevo.base.Sampler` for querying the LLM.
    - **evaluator** (Evaluator): An instance of `alevo.base.Evaluator` to calculate the score of the generated function.
    - **profiler** (FunSearchProfiler, optional): An instance of `alevo.method.funsearch.FunSearchProfiler`. Pass `None` if profiling is not needed.
    - **config** (ProgramsDatabaseConfig, optional): An instance of `alevo.method.funsearch.config.ProgramDatabaseConfig`. Defaults to a new instance.
    - **max_sample_nums** (int, optional): Maximum number of functions to evaluate. Defaults to 20.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **samples_per_prompt** (int, optional): Number of samples to generate per prompt. Defaults to 4.
    - **valid_only** (bool, optional): If set to `True`, only valid functions are registered. Defaults to `False`.
    - **kwargs**: Additional arguments passed to `alevo.base.SecureEvaluator`.

.. important::
    **template_program** The template program in must be a valid algorithm (obtain a valid score during evaluation).


Methods
-------

.. method:: run()

    Starts the function search optimization process. If `resume_mode` is `False`, it initializes the algorithm by evaluating the template program and then starts sampling using multiple threads.

Private Methods
---------------

.. method:: _sample_evaluate_register()

    Continuously samples new functions, evaluates them, and registers the results until the maximum sample count is reached.

Attributes
----------

- **_template_program_str** (str): The string representation of the template program.
- **_max_sample_nums** (int | None): The maximum number of samples to evaluate.
- **_debug_mode** (bool): Indicates if debug mode is enabled.
- **_resume_mode** (bool): Indicates if resume mode is enabled.
- **_function_to_evolve** (Function): The function that will be evolved.
- **_database** (ProgramsDatabase): The database for managing program instances.
- **_sampler** (Sampler): The sampler instance used for sampling.
- **_evaluator** (Evaluator): The evaluator instance used for evaluation.
- **_profiler** (FunSearchProfiler): The profiler instance, if used.
- **_samples_per_prompt** (int): Number of samples to generate per prompt.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_evaluation_executor** (concurrent.futures.Executor): The executor for parallel evaluation.
- **_sampler_threads** (List[Thread]): The list of threads used for sampling.

Exceptions
----------

- **RuntimeError**: Raised if the score of the template function is `None`.
