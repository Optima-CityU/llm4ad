HillClimb
===============

The `HillClimb` class implements a hill climbing algorithm to optimize a given program using sampling and evaluation.

Usage
-----

To use the `HillClimb` class, you need to initialize it with the required parameters and then call the `run` method to start the optimization process.

Constructor
-----------

.. class:: HillClimb

    .. rubric:: Parameters

    - **template_program** (str): The seed program as the initial function of the run. The program should be executable and include necessary imports and definitions.
    - **sampler** (Sampler): An instance of `alevo.base.Sampler` for querying the LLM.
    - **evaluator** (Evaluator): An instance of `alevo.base.Evaluator` to calculate the score of the generated function.
    - **profiler** (HillClimbProfiler, optional): An instance of `alevo.method.hillclimb.HillClimbProfiler`. Pass `None` if profiling is not needed.
    - **max_sample_nums** (int, optional): Maximum number of functions to evaluate. Defaults to 20.
    - **num_samplers** (int, optional): Number of sampler threads. Defaults to 4.
    - **num_evaluators** (int, optional): Number of evaluator threads. Defaults to 4.
    - **valid_only** (bool, optional): If set to `True`, only valid functions are registered. Defaults to `False`.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **initial_sample_num** (int, optional): Initial sample count. Defaults to `None`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `alevo.base.SecureEvaluator`.

.. important::
    **template_program** The template program in must be a valid algorithm (obtain a valid score during evaluation).

Methods
-------

.. method:: run()

    Start the hill climbing optimization process. If `resume_mode` is `False`, it initializes the algorithm and then starts sampling using multiple threads.

Private Methods
---------------

.. method:: _init()

    Initializes the hill climbing process by evaluating the template program and registering it.

.. method:: _get_prompt() -> str

    Generates the prompt for the next sampling iteration.

.. method:: _sample_evaluate_register()

    Continuously samples new functions, evaluates them, and updates the best function found until the maximum sample count is reached.

Attributes
----------

- **_template_program_str** (str): The string representation of the template program.
- **_max_sample_nums** (int | None): The maximum number of samples to evaluate.
- **_valid_only** (bool): Indicates if only valid functions should be registered.
- **_debug_mode** (bool): Indicates if debug mode is enabled.
- **_resume_mode** (bool): Indicates if resume mode is enabled.
- **_function_to_evolve** (Function): The function that will be evolved.
- **_best_function_found** (Function): The best function found during the optimization process.
- **_sampler** (Sampler): The sampler instance used for sampling.
- **_evaluator** (Evaluator): The evaluator instance used for evaluation.
- **_profiler** (HillClimbProfiler): The profiler instance, if used.
- **_evaluation_executor** (concurrent.futures.Executor): The executor for parallel evaluation.
- **_sampler_threads** (List[Thread]): The list of threads used for sampling.

Exceptions
----------

- **RuntimeError**: Raised if the score of the template function is `None`.
