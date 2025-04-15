from __future__ import annotations
import re
import copy
from ..code import TextFunctionProgramConverter, Function

from typing import Any, List, Callable

import dataclasses
@dataclasses.dataclass
class KERFunction:
    """A parsed Kernel function."""
    includes: str
    func_title: str
    name: str
    body: str
    score: Any | None = None
    evaluate_time: float | None = None
    sample_time: float | None = None
    def __str__(self) -> str:
        if len(self.body) == 0:
            func_content = ""
            func_content += f"{self.includes}\n"
            func_content += f"{self.func_title}\n"
            return func_content
        else:
            return self.body

@dataclasses.dataclass(frozen=True)
class KERProgram:
    preface: str
    functions: list[KERFunction]
    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        return program

class _KERProgramVisitor:
    def __init__(self):
        self.functions = []

    @classmethod
    def _find_func_title(cls, program_str, operation_name):
        code_title_re = re.compile(r"\n.*\s+" + operation_name + r"\s*[\s\S]*?{")
        code_title = code_title_re.findall(program_str)
        return code_title[0]

    @classmethod
    def visit_FunctionDef(cls, code_str: str):
        operation_name = cls._find_operation_name(code_str)
        includes, rest = cls._extract_includes(code_str)
        func_title = cls._find_func_title(code_str, operation_name)
        ker_function = KERFunction(
            includes=includes,
            name=operation_name,
            body=code_str,
            func_title=func_title,
        )
        return ker_function

    @staticmethod
    def _find_operation_name(cuda_code: str) -> str:
        pattern = r'm\.def\([^,]+,\s*&(\w+)'
        matches = re.findall(pattern, cuda_code)
        if matches:
            return matches[0]

    @staticmethod
    def _extract_includes(code_str: str) -> str:
        includes = ''
        rest = ''
        for line in code_str.split('\n'):
            if line.startswith('#include'):
                includes += line + '\n'
            else:
                rest += line + '\n'

        return includes, rest


class KERTextFunctionProgramConverter:
    @classmethod
    def text_to_function_py(cls, python_code: str) -> [Function, KERFunction] | None:
        python_program = TextFunctionProgramConverter.text_to_program(python_code)
        for each_python_func in python_program.functions:
            if each_python_func.name == 'module_fn':
                python_func = each_python_func
                break
        return python_func

    @classmethod
    def text_to_function(cls, cpp_code: str) -> [Function, KERFunction] | None:
        try:
            cpp_program = cls.text_to_program(cpp_code)
            return cpp_program.functions[0]
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def text_to_program(cls, program_str: str) -> KERProgram | None:
        """Returns Program object by parsing input text using Python AST.
        """
        try:
            visitor = _KERProgramVisitor()
            func = visitor.visit_FunctionDef(program_str)

            # We assume that the program is composed of some preface (e.g. imports,
            # classes, assignments, ...) followed by a sequence of functions.
            return KERProgram(preface='', functions=[func])
        # catch all exceptions
        except Exception as e:
            return None

    @classmethod
    def program_to_function(cls, program: KERProgram) -> KERFunction:
        return program.functions[0]

    @classmethod
    def function_to_program(cls, function: str | Function, template_program: str | KERProgram) -> KERProgram | None:
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