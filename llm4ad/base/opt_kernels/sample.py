# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import re
import ast
import copy
from abc import abstractmethod
from typing import Any, List

from .code import KERProgram, KERFunction, KERTextFunctionProgramConverter


class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, *, do_auto_trim=True, debug_mode=False):
        self.do_auto_trim = do_auto_trim
        self.debug_mode = debug_mode

    @abstractmethod
    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """Returns a predicted continuation of `prompt`.
        -For example, the response content of the LLM is:
        ------------------------------------------------------------------------------------------------------------------
        Here is the function.
        def priority_v2(..., ...) -> Any:
            a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        This function is going to ..., and returns ...[Descriptions by LLM]
        ------------------------------------------------------------------------------------------------------------------
        """
        pass

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        """Returns multiple predicted continuations of `prompt`.
        """
        return [self.draw_sample(p, *args, **kwargs) for p in prompts]


class CPPSampleTrimmer:
    def __init__(self, llm: LLM):
        self._llm = llm

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """Get a sample based on the provided 'LLM' instance.
        If the inner sampler sets 'auto_trim' to True, trim anything before the function body.
        """
        generated_code = self._llm.draw_sample(prompt, *args, **kwargs)
        if self._llm.do_auto_trim:
            generated_code = self.__class__.auto_trim(generated_code)
        return generated_code

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        """Get samples based on the provided 'Sampler' instance.
        If the inner sampler sets 'auto_trim' to True, trim anything before the function body.
        """
        ret = self._llm.draw_samples(prompts, *args, **kwargs)
        if self._llm.do_auto_trim:
            ret = [self.__class__.auto_trim(code) for code in ret]
        return ret

    @classmethod
    def _check_indent_if_code_completion(cls, generated_code: str) -> bool:
        """Judge if the content is generated through code completion model or instruct model.
        """
        generated_code = generated_code.strip('\n')
        line = generated_code.splitlines()[0]
        if line.startswith('\t'):
            return True
        if line.startswith(' ' * 2):
            return True
        if line.startswith(' ' * 4):
            return True
        return False

    @classmethod
    def trim_preface_of_function(cls, generated_code: str):
        # C++ code start with #include, end with last }
        re_pattern = r'(#include.*)\n'
        matched_code = re.findall(re_pattern, generated_code, re.DOTALL)

        return matched_code[0]

    @classmethod
    def auto_trim(cls, generated_code: str) -> str:
        """Automatically trim the preface of the generated content.
        """
        is_code_complete = cls._check_indent_if_code_completion(generated_code)
        if is_code_complete:
            return generated_code
        generated_code = cls.trim_preface_of_function(generated_code)
        return generated_code

    @classmethod
    def sample_to_function(cls, generated_code: str, template_program: str | KERProgram) -> KERFunction | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        Please note that the modified Function instance is not executable,
        as it lacks 'import ...' statements.
        """
        program = cls.sample_to_program(generated_code, template_program)
        if program is None:
            return None
        return KERTextFunctionProgramConverter.program_to_function(program)

    @classmethod
    def sample_to_program(cls, generated_code: str, template_program: str | KERProgram) -> KERProgram | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        """
        try:
            # convert program to Program instance
            if isinstance(template_program, str):
                template_program = KERTextFunctionProgramConverter.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)
            template_program.functions[0].body = generated_code
            if template_program.functions[0].body == '' or template_program.functions[0].body is None:
                return None
            # ------------------------------------------------------------------------------------------------
            return template_program
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def remove_docstrings(cls, func: KERFunction | str):
        func_ = copy.deepcopy(func)
        func_ = KERTextFunctionProgramConverter.text_to_function(str(func_))  # convert to Function instance
        docstring = func_.docstring
        while not (docstring == "" or docstring is None):
            func_.docstring = ""
            func_str = str(func_)
            func_ = KERTextFunctionProgramConverter.text_to_function(func_str)
            docstring = func_.docstring

        if isinstance(func, KERFunction):
            for key, value in func.__dict__.items():
                if key != 'docstring' and key != 'body':
                    setattr(func_, key, value)
            return func_
        else:
            return str(func_)


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line
