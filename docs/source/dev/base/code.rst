base.code
================

This module contains two dataclasses, `Function` and `Program`, designed for parsing and representing Python code. These classes provide methods to manage function details, parse code structure, and retrieve specific functions from a program.

Class Definitions
-----------------

.. class:: Function

   Represents a parsed Python function.

   **Attributes**:

   - **name** (str): Name of the function.

   - **args** (str): Function arguments as a string.

   - **body** (str): The function body (indented).

   - **return_type** (str | None): Return type annotation of the function, if present.

   - **docstring** (str | None): Optional docstring describing the function.

   - **score** (Any | None): Optional attribute for evaluating or ranking functions.

   - **evaluate_time** (float | None): Time taken to evaluate the function, if applicable.

   - **sample_time** (float | None): Time taken to sample or retrieve the function, if applicable.

.. class:: Program

   Represents a parsed Python program consisting of a preface and a list of `Function` objects.

   **Attributes**:

   - **preface** (str): Code content before the first function is defined.

   - **functions** (list[Function]): List of `Function` objects representing each function in the program.

   **Methods**:

   - **find_function_index(function_name: str) -> int**: Finds the index of a function in `functions` by its name. Raises a `ValueError` if the function name does not exist or if it exists multiple times in the program.

   - **get_function(function_name: str) -> Function**: Retrieves a `Function` instance by its name using the `find_function_index` method.
