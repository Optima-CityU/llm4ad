base.sample
==================

Class Definitions
-----------------

.. class:: Sampler

    An abstract class designed for predicting continuations of provided source code using a language model.

    Constructor Parameters
    -----------------------
    - **do_auto_trim** (bool): If `True`, automatically trims any leading content before the function body from generated samples.

    Methods
    -------
    - **draw_sample(self, prompt: str | Any, *args, **kwargs) -> str**:
        Abstract method that should be implemented to return a predicted continuation of the provided `prompt`. This method is expected to yield code samples based on input prompts.

        - **Example**:
        .. code-block:: python

            # Sample output
            Here is the function.
            def priority_v2(..., ...) -> Any:
                ...

    - **draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]**:
        Returns multiple predicted continuations based on a list of prompts by calling `draw_sample` for each one.

.. class:: SamplerTrimmer

    A utility class that wraps around a `Sampler` instance, providing functionality to trim unnecessary preamble content from generated code.

    Constructor Parameters
    -----------------------
    - **sampler** (Sampler): An instance of the `Sampler` class that will be used for generating code samples.

    Methods
    -------
    - **draw_sample(self, prompt: str | Any, *args, **kwargs) -> str**:
        Calls the `Sampler` instance's `draw_sample` method and trims the generated code if `do_auto_trim` is enabled.

    - **draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]**:
        Similar to `draw_sample`, but for a list of prompts, returning trimmed samples.

    - **@classmethod _check_indent_if_code_completion(cls, generated_code: str) -> bool**:
        Checks if the generated code is likely from a code completion model by inspecting the indentation of the first line.

    - **@classmethod trim_preface_of_function(cls, generated_code: str) -> str**:
        Trims any descriptions or symbols before the actual function body.

        - **Example**:
            Input:

            .. code-block:: python

                This is the optimized function ...
                def priority_v2(...) -> ...:
                    a = random.random()
                    return a * a

            Output:

            .. code-block:: python

                a = random.random()
                return a * a

    - **@classmethod auto_trim(cls, generated_code: str) -> str**:
        Automatically trims the preface of the generated code if necessary.

    - **@classmethod sample_to_function(cls, generated_code: str, template_program: str | Program) -> Function | None**:
        Converts the trimmed generated code into a `Function` instance based on a provided template program. Returns `None` if conversion fails.

    - **@classmethod sample_to_program(cls, generated_code: str, template_program: str | Program) -> Program | None**:
        Converts the generated code to a `Program` instance, using a template program as a base. Handles possible conversion errors.

    - **@classmethod trim_function_body(cls, generated_code: str) -> str | None**:
        Extracts and returns the body of the generated function while trimming any extraneous content that follows it.

    - **@classmethod remove_docstrings(cls, func: Function | str)**:
        Removes docstrings from a function instance, ensuring that the returned function does not contain any unnecessary documentation.
