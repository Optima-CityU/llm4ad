from __future__ import annotations
import re
from clang.cindex import Index, CursorKind, TranslationUnit, Config
from ..code import TextFunctionProgramConverter, Function

import dataclasses
@dataclasses.dataclass
class KERFunction:
    """A parsed Kernel function."""
    name: str
    body: str
    def __str__(self) -> str:
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
    def find_returns_in_child(cls, node):
        if node.kind == CursorKind.RETURN_STMT:
            print(f"Found return statement at {node.location}")

        # 递归遍历子节点
        for child in node.get_children():
            cls.find_returns_in_child(child)

    @classmethod
    def visit_FunctionDef(cls, code_str: str):
        operation_name = cls._find_operation_name(code_str)
        ker_function = KERFunction(name=operation_name, body=code_str)
        return ker_function

    @staticmethod
    def _find_operation_name(cuda_code: str) -> str:
        pattern = r'm\.def\([^,]+,\s*&(\w+)'
        matches = re.findall(pattern, cuda_code)
        if matches:
            return matches[0]


class KERTextFunctionProgramConverter:
    @classmethod
    def text_to_function(cls, cpp_code: str, python_code: str) -> [Function, KERFunction] | None:
        try:
            # python_func = TextFunctionProgramConverter.text_to_function(python_code)
            python_program = TextFunctionProgramConverter.text_to_program(python_code)
            cpp_program = cls.text_to_program(cpp_code)
            for each_python_func in python_program.functions:
                if each_python_func.name == 'module_fn':
                    python_func = each_python_func
                    break
            return cpp_program.functions[0], python_func
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