RandSample
================

The `RandSample` class implements a random sampling strategy for evaluating functions based on a given template program.

Usage
-----

To utilize the `RandSample` class, initialize it with the necessary parameters and call the `run` method to start the sampling and evaluation process.

Constructor
-----------

.. class:: RandSample

    .. rubric:: Parameters

    - **template_program** (str): The seed program as the initial function. Should be executable, including imports, definitions, and body.
    - **sampler** (Sampler): An instance of `alevo.base.Sampler` for querying the LLM.
    - **evaluator** (Evaluator): An instance of `alevo.base.Evaluator` to calculate the scores of generated functions.
    - **profiler** (RandSampleProfiler, optional): An instance of `alevo.method.randsample.RandSampleProfiler`. Pass `None` if profiling is not needed.
    - **max_sample_nums** (int | None, optional): Maximum number of samples to evaluate. Defaults to 20.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **initial_sample_num** (int | None, optional): Initial count of samples evaluated. Defaults to `None`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str, optional): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **valid_only** (bool, optional): If set to `True`, only valid functions are registered. Defaults to `False`.
    - **num_samplers** (int, optional): Number of sampling threads. Defaults to 4.
    - **num_evaluators** (int, optional): Number of evaluation threads. Defaults to 4.
    - **kwargs**: Additional arguments passed to `alevo.base.SecureEvaluator`.

.. important::
    **template_program** The template program in must be a valid algorithm (obtain a valid score during evaluation).

Methods
-------

.. method:: run()

    Starts the sampling and evaluation process. If `resume_mode` is `False`, it evaluates the template program and initializes the profiler.

Private Methods
---------------

.. method:: _get_prompt() -> str

    Generates a prompt based on the template program and the function to be evolved.

.. method:: _sample_evaluate_register()

    Repeatedly samples functions, evaluates them, and registers the results with the profiler.

Attributes
----------

- **_template_program_str** (str): The string representation of the template program.
- **_function_to_evolve** (Function): The function that will be evolved.
- **_sampler** (SamplerTrimmer): The sampler instance used for sampling.
- **_evaluator** (Evaluator): The evaluator instance used for evaluation.
- **_profiler** (RandSampleProfiler): The profiler instance, if used.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_prompt_content** (str): The generated prompt content for sampling.

Exceptions
----------

- **RuntimeError**: Raised if the score of the template function is `None` during initialization.
