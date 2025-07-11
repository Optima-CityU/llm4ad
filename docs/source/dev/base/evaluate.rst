base.evaluate
====================

This module provides two main classes: `Evaluator` and `SecureEvaluator`. These classes enable the secure and configurable evaluation of generated code, incorporating optional optimizations and safety features like timeout handling and protected division.

Class Definitions
-----------------

.. class:: Evaluator

    An abstract base class for evaluating Python code with customizable options to handle division safety, random seed setting, and code execution.

    **Constructor Parameters**:

    - **use_numba_accelerate** (bool): If `True`, applies the `@numba.jit(nopython=True)` decorator for function acceleration.

    - **use_protected_div** (bool): If `True`, modifies division operations to use a protected division method, avoiding division by zero.

    - **protected_div_delta** (float): Delta value for protected division (default is `1e-5`).

    - **random_seed** (int | None): If specified, sets a random seed at the start of the function for reproducibility.

    - **timeout_seconds** (int | float): Time limit for code evaluation, in seconds.

    - **exec_code** (bool): If `True`, uses `exec()` to compile the code and generate a callable function.

    - **safe_evaluate** (bool): If `True`, evaluates code in a separate process, enabling timeout-based termination.

    - **daemon_eval_process** (bool): If `True`, sets the evaluation process as a daemon, preventing additional processes within the evaluation function.

    **Example Modification**:
    When `use_numba_accelerate=True`, `use_protected_div=True`, and `random_seed=2024`, an input function:

    .. code-block:: python

        import numpy as np

        def f(a, b):
           a = np.random.random()
           return a / b

    Will be modified to:

    .. code-block:: python

        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f():
           np.random.seed(2024)
           a = np.random.random()
           return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
           return a / (b + delta)

    **Method**:
    - **evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None**: Abstract method for evaluating a function. Takes in the function's string representation (`program_str`) and a callable (`callable_func`). Returns the function's fitness value. This method is intended to be overridden in subclasses.

.. class:: SecureEvaluator

    Wraps an `Evaluator` instance to ensure safe execution and modify program code as needed, with debug mode support and configurable process handling for enhanced security.

    **Constructor Parameters**:

    - **evaluator** (Evaluator): An instance of `Evaluator` or its subclass for executing code.

    - **debug_mode** (bool): If `True`, provides debug outputs to help trace program evaluation steps.

    - **fork_proc** (str | bool): Determines how new processes are generated:
        - `'auto'`: Automatically uses `'fork'` for macOS and Linux.
        - `True`: Forces `'fork'` for process creation.
        - `False`: Forces `'spawn'` for process creation.

    **Method**:

    - **evaluate_program(self, program: str | Program, **kwargs)**: Safely evaluates the program, modifying it as needed, and returns the result. Manages timeouts and process termination if `safe_evaluate` is `True`.

    - **evaluate_program_record_time(self, program: str | Program, **kwargs)**: Records and returns the time taken to evaluate the program.

This setup ensures a controlled evaluation environment, handling exceptions and logging when `debug_mode` is enabled.
