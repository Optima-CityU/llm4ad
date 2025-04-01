"""
This file implements 2 classes representing unities of MATLAB code:

- Function, containing all the information we need about MATLAB functions: name,
  inputs, outputs, body and optionally a help text (documentation).

- Program, which contains a code preface (which could be global variables,
  addpath statements, etc.) and a list of Functions.

- For example, a MATLAB function is shown below:
--------------------------------------------------------------------------------------------
function [c] = func(a, b)
    % Add two matrices with a weight
    global WEIGHT;
    b = b + WEIGHT;
    c = a + b;
end
--------------------------------------------------------------------------------------------

- A complete MATLAB program would be:
--------------------------------------------------------------------------------------------
global WEIGHT;
WEIGHT = 10;

function [c] = func(a, b)
    % Add two matrices with a weight
    global WEIGHT;
    b = b + WEIGHT;
    c = a + b;
end
--------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from typing import Any, List, Callable


@dataclasses.dataclass
class MatlabFunction:
    """A parsed MATLAB function."""

    algorithm = ''
    name: str
    inputs: str  # Changed from args to inputs
    outputs: str  # Added for MATLAB output parameters
    body: str
    help_text: str | None = None  # Changed from docstring to help_text
    score: Any | None = None
    evaluate_time: float | None = None
    sample_time: float | None = None

    def __str__(self) -> str:
        # Format for MATLAB function
        function = f'function {self.outputs} = {self.name}({self.inputs})\n'
        if self.help_text:
            # MATLAB uses % for comments
            help_lines = self.help_text.split('\n')
            function += '\n'.join(f'    % {line.strip()}' for line in help_lines) + '\n'

        function += self.body # + '\nend\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`
        if name == 'body':
            value = value.strip('\n')
        # Ensure help_text is properly formatted
        if name == 'help_text' and value is not None:
            value = value.strip('%').strip()
        super().__setattr__(name, value)

    def __eq__(self, other: MatlabFunction):
        assert isinstance(other, MatlabFunction)
        return (self.name == other.name and
                self.inputs == other.inputs and
                self.outputs == other.outputs and
                self.body == other.body)


@dataclasses.dataclass(frozen=True)
class MatlabProgram:
    """A parsed MATLAB program."""

    preface: str
    functions: list[MatlabFunction]

    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        return program

    def find_function_index(self, function_name: str) -> int:
        """Returns the index of input function name."""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f'function {function_name} does not exist in program:\n{str(self)}'
            )
        if count > 1:
            raise ValueError(
                f'function {function_name} exists more than once in program:\n'
                f'{str(self)}'
            )
        index = function_names.index(function_name)
        return index

    def get_function(self, function_name: str) -> MatlabFunction:
        index = self.find_function_index(function_name)
        return self.functions[index]


class _MatlabProgramVisitor:
    """Parses MATLAB code to collect all required information to produce a `Program`.
    Note: This is a simplified parser for MATLAB code structure.
    """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()
        self._preface: str = ''
        self._functions: list[MatlabFunction] = []
        self._current_function: str | None = None

    def parse(self) -> MatlabProgram:
        """Parse MATLAB code and return Program instance."""
        current_function = None
        function_start = None
        help_text = []

        for i, line in enumerate(self._codelines):
            # line = line.strip()

            # Check for function definition
            if line.startswith('function'):
                if current_function is None:
                    # This is the first function - everything before is preface
                    self._preface = '\n'.join(self._codelines[:i])

                # Parse function signature
                # Example: function [out1, out2] = funcname(in1, in2)
                function_def = line[8:].strip()  # Remove 'function' keyword
                outputs = ''
                if '[' in function_def:
                    outputs = function_def[function_def.find('[') + 1:function_def.find(']')]
                    function_def = function_def[function_def.find(']') + 1:]
                else:
                    outputs = function_def[:function_def.find('=')].strip()

                name_and_inputs = function_def.split('=')[-1].strip()
                name = name_and_inputs[:name_and_inputs.find('(')].strip()
                inputs = name_and_inputs[name_and_inputs.find('(') + 1:name_and_inputs.find(')')].strip()

                current_function = {
                    'name': name,
                    'inputs': inputs,
                    'outputs': outputs,
                    'start_line': i
                }
                help_text = []

            # Collect help text (comments right after function definition)
            elif current_function and line.strip().startswith('%'):
                help_text.append(line.strip()[1:].strip())

            # Check for function end
            elif line == 'end' and current_function:
                body = '\n'.join(self._codelines[current_function['start_line'] + 1:i+1])

                self._functions.append(MatlabFunction(
                    name=current_function['name'],
                    inputs=current_function['inputs'],
                    outputs=current_function['outputs'],
                    body=body,
                    help_text='\n'.join(help_text) if help_text else None
                ))

                current_function = None
                help_text = []

        return MatlabProgram(preface=self._preface, functions=self._functions)


class TextMatlabFunctionProgramConverter:
    """Convert text to Program instance and Function instance for MATLAB code."""

    @classmethod
    def text_to_program(cls, program_str: str) -> MatlabProgram | None:
        """Returns Program object by parsing input MATLAB text."""
        try:
            visitor = _MatlabProgramVisitor(program_str)
            return visitor.parse()
        except:
            return None

    @classmethod
    def text_to_function(cls, program_str: str) -> MatlabFunction | None:
        """Returns Function object by parsing input MATLAB text."""
        try:
            program = cls.text_to_program(program_str)
            if len(program.functions) != 1:
                raise ValueError(f'Only one function expected, got {len(program.functions)}'
                                 f':\n{program.functions}')
            return program.functions[0]
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def function_to_program(cls, function: str | MatlabFunction, template_program: str | MatlabProgram) -> MatlabProgram | None:
        try:
            # convert function to Function instance
            if isinstance(function, str):
                function = cls.text_to_function(function)
            else:
                function = copy.deepcopy(function)

            # convert template_program to Program instance
            if isinstance(template_program, str):
                template_program = cls.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)

            # assert that a program have one function
            if len(template_program.functions) != 1:
                raise ValueError(f'Only one function expected, got {len(template_program.functions)}'
                                 f':\n{template_program.functions}')

            # replace the function body with the new function body
            template_program.functions[0].body = function.body
            return template_program
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def program_to_function(cls, program: str | MatlabProgram) -> MatlabFunction | None:
        try:
            # convert program to Program instance
            if isinstance(program, str):
                program = cls.text_to_program(program)
            else:
                program = copy.deepcopy(program)

            # assert that a program have one function
            if len(program.functions) != 1:
                raise ValueError(f'Only one function expected, got {len(program.functions)}'
                                 f':\n{program.functions}')

            # return the function
            return program.functions[0]
        except ValueError as value_err:
            raise value_err
        except:
            return None
