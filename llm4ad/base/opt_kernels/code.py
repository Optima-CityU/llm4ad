from __future__ import annotations
import re
from clang.cindex import Index, CursorKind, TranslationUnit, Config

import dataclasses
@dataclasses.dataclass
class KERFunction:
    """A parsed Kernel function."""
    name: str

@dataclasses.dataclass(frozen=True)
class KERProgram:
    preface: str
    functions: list[KERFunction]

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
        Config.set_library_file(r"C:\\Program Files\\LLVM\\bin\\libclang.dll")  # Windows
        index = Index.create()
        translation_unit = index.parse('tmp.cpp', args=['-std=c++11'],
                                       unsaved_files=[('tmp.cpp', code_str)])

        cursor = translation_unit.cursor

        args = []
        return_type = None
        for each_child in cursor.get_children():
            if each_child.kind == CursorKind.FUNCTION_DECL:
                if each_child.spelling == operation_name:
                    # pick the input arguments
                    for func_child in each_child.get_children():
                        if func_child.kind == CursorKind.PARM_DECL:
                            args.append(func_child.spelling)
                        cls.find_returns_in_child(func_child)

        functions = []
        variables = []

        def print_ast(node, indent=0):
            # 打印当前节点的信息
            print('  ' * indent + f'{node.kind}: {node.spelling} [{node.location}]')

            # 递归打印子节点
            for child in node.get_children():
                print_ast(child, indent + 1)

            # 打印 AST
        print_ast(translation_unit.cursor)
        return functions, variables

    @staticmethod
    def _find_operation_name(cuda_code: str) -> str:
        pattern = r'm\.def\([^,]+,\s*&(\w+)'
        matches = re.findall(pattern, cuda_code)
        if matches:
            return matches[0]


class KERTextFunctionProgramConverter:
    @classmethod
    def text_to_function(cls, cpp_code: str, python_code: str) -> KERFunction | None:
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
    def text_to_program(cls, program_str: str) -> KERProgram | None:
        """Returns Program object by parsing input text using Python AST.
        """
        try:
            visitor = _KERProgramVisitor()
            funcs, variables = visitor.visit_FunctionDef(program_str)

            # We assume that the program is composed of some preface (e.g. imports,
            # classes, assignments, ...) followed by a sequence of functions.
            return KERProgram(preface='', functions=[program_str])
        # catch all exceptions
        except Exception as e:
            return None