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
    def find_function_definition(cls, node, func_name, code_str):
        if node.kind == CursorKind.FUNCTION_DECL and node.spelling == func_name:
            start_line = node.extent.start.line

            def get_line_from_code_str(code_str, line_number):
                """
                从 code_str 中获取指定行的内容。
                """
                lines = code_str.splitlines()
                if 1 <= line_number <= len(lines):
                    return lines[line_number - 1].strip()  # 行号从 1 开始
                return None

            decl_line = get_line_from_code_str(code_str, start_line)
            print(decl_line)
            return decl_line
        for child in node.get_children():
            cls.find_function_definition(child, func_name, code_str)

    @classmethod
    def visit_FunctionDef(cls, code_str: str):
        operation_name = cls._find_operation_name(code_str)
        # Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
        # index = Index.create()
        # translation_unit = index.parse('tmp.cpp', args=['-std=c++11'],
        #                                unsaved_files=[('tmp.cpp', code_str)])
        # cursor = translation_unit.cursor

        # func_name_line = cls.find_function_definition(cursor, operation_name, code_str)
        includes, rest = cls._extract_includes(code_str)
        ker_function = KERFunction(name=operation_name, body=code_str)
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