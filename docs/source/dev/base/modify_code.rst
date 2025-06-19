base.modify_code
=======================

Class Definitions
-----------------

.. class:: ModifyCode

   The `ModifyCode` class provides a collection of methods for programmatically modifying Python code strings. It includes functionalities for adding decorators, import statements, modifying function definitions, and managing function calls.

   Methods
   -------

   - **add_decorator**

     Adds a decorator to a specified function within the provided program.

     - **Parameters**:

       - **program**: The source code as a string.

       - **function_name**: The name of the function to be decorated.

       - **decorator_name**: A string or list of strings representing the decorator.

       - **decorator_args**: Optional arguments for the decorator, can be a list of strings or tuples.

     - **Returns**: The modified code with the decorator added.

     - **Examples**:
         Adding a simple decorator:

         .. code-block:: python

             ModifyCode.add_decorator(program, 'f', 'torch.jit.script')


         Adding a decorator with arguments:

         .. code-block:: python

             ModifyCode.add_decorator(program, 'f', ['numba', 'jit'], [('nopython', True)])

   - **add_import_package_statement**

     Adds an import statement for a package to the program.

     - **Parameters**:

       - **program**: The source code as a string.

       - **package_name**: The name of the package to import.

       - **as_name**: Optional alias for the package.

       - **check_imported**: If `True`, checks if the import statement already exists.

     - **Returns**: The modified code with the import statement added.

   - **add_numpy_random_seed_to_func**

     Inserts a random seed setting at the beginning of a specified function.

     - **Parameters**:

       - **program**: The source code as a string.

       - **func_name**: The name of the function to modify.

       - **seed**: The random seed to set.

     - **Returns**: The modified code with the random seed added.

   - **replace_div_with_protected_div**

     Replaces division operations with a protected division function to prevent division by zero errors.

     - **Parameters**:

       - **program**: The source code as a string.

       - **delta**: A small value added to the denominator.

       - **numba_accelerate**: If `True`, applies Numba acceleration.

       - **return_div_func_name**: If `True`, returns the name of the protected division function.

     - **Returns**: The modified code or a tuple containing the modified code and the division function name.

   - **add_np_random_seed_below_numpy_import**

     Adds a random seed setting immediately after importing NumPy.

     - **Parameters**:

       - **program**: The source code as a string.

       - **seed**: The random seed to set.

     - **Returns**: The modified code with the random seed added.

   - **add_numba_decorator**

     Adds the Numba `@jit` decorator to specified functions to optimize performance.

     - **Parameters**:

       - **program**: The source code as a string.

       - **function_name**: The name of the function or a list of functions to decorate.

     - **Returns**: The modified code with the Numba decorator added.

   - **rename_function**

     Renames occurrences of a function within the provided code.

     - **Parameters**:

       - **code**: The source code as a string.

       - **source_name**: The current name of the function.

       - **target_name**: The new name for the function.

     - **Returns**: The modified code with the function name changed.

   - **get_functions_name**

     Extracts and returns a set of all function names defined in the provided code.

     - **Parameters**:
       - **code**: The source code as a string.

     - **Returns**: A set of function names.

   - **yield_decorated**

     Yields the names of functions decorated with a specified decorator.

     - **Parameters**:

       - **code**: The source code as a string.

       - **module**: The module of the decorator.

       - **name**: The name of the decorator.

     - **Returns**: An iterator yielding the names of decorated functions.
