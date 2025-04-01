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

import copy
import re
from abc import abstractmethod
from typing import Any, List

from .code_matlab import MatlabProgram, MatlabFunction, TextMatlabFunctionProgramConverter


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
        function y = priority_v2(x1, x2)
            % Function description
            a = [1, 2, 3];
            if length(a) > 2
                y = a ./ sum(a);
            else
                y = a ./ mean(a);
            end
        end
        This function is going to ..., and returns ...[Descriptions by LLM]
        ------------------------------------------------------------------------------------------------------------------
        """
        pass

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        """Returns multiple predicted continuations of `prompt`.
        """
        return [self.draw_sample(p, *args, **kwargs) for p in prompts]


class SampleTrimmerMat:
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
        """Trim the redundant descriptions/symbols/function declaration BEFORE the function body.
        Example of a generated content from an LLM:
        --------------------------------------------------------------------------
        This is the optimized function ...

        function y = priority_v2(x1, x2)
            a = rand();
            y = a * a;
        end

        This function aims to ...
        --------------------------------------------------------------------------
        Example return of this function:
        --------------------------------------------------------------------------
            a = rand();
            y = a * a;

        This function aims to ...
        --------------------------------------------------------------------------
        """
        lines = generated_code.splitlines()
        func_body_start = 0
        func_body_end = len(lines)

        # Find function start and end
        for i, line in enumerate(lines):
            # line = line.strip()
            if line.startswith('function'):
                func_body_start = i
            elif line == 'end' and i > func_body_start:
                func_body_end = i
                break

        if func_body_start < func_body_end:
            return '\n'.join(lines[func_body_start:func_body_end+1])
        return generated_code

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
    def sample_to_function(cls, generated_code: str, template_program: str | MatlabProgram) -> MatlabFunction | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        """
        program = cls.sample_to_program(generated_code, template_program)
        if program is None:
            return None
        return TextMatlabFunctionProgramConverter.program_to_function(program)

    @classmethod
    def sample_to_program(cls, generated_code: str, template_program: str | MatlabProgram) -> MatlabProgram | None:
        """Convert the generated content (with redundant component)
        to a Program instance. If the convert fails, return None.
        """
        try:
            generated_code = cls.trim_function_body(generated_code)
            # convert program to Program instance
            if isinstance(template_program, str):
                template_program = TextMatlabFunctionProgramConverter.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)

            # store a help text copy
            help_text_copy = template_program.functions[0].help_text

            # replace the function body with the generated body
            template_program.functions[0].body = generated_code

            # Remove redundant help texts and restore original
            template_program.functions[0] = cls.remove_help_texts(template_program.functions[0])
            if template_program.functions[0].body == '' or template_program.functions[0].body is None:
                return None

            template_program.functions[0].help_text = help_text_copy

            return template_program
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def trim_function_body(cls, generated_code: str) -> str | None:
        """Extracts the body of the generated MATLAB function, trimming anything after it.
        """
        try:
            if not generated_code:
                return ''

            lines = generated_code.splitlines()
            body_lines = []
            in_function = False

            for line in lines:
                # line = line.strip()
                if line.startswith('function'):
                    in_function = True
                    continue
                elif line == 'end' and in_function:
                    body_lines.append(line)
                    break
                elif in_function:
                    body_lines.append(line)

            return '\n'.join(body_lines) + '\n\n' if body_lines else None

        except:
            return None

    @classmethod
    def remove_help_texts(cls, func: MatlabFunction | str) -> MatlabFunction:
        """Remove help texts (comments) from MATLAB function while preserving the original structure.
        """
        if isinstance(func, str):
            func = TextMatlabFunctionProgramConverter.text_to_function(func)

        func_copy = copy.deepcopy(func)
        lines = func_copy.body.splitlines()
        clean_lines = []

        for line in lines:
            # Remove comment lines but keep empty lines for structure
            if not line.strip().startswith('%'):
                clean_lines.append(line)

        func_copy.body = '\n'.join(clean_lines)
        func_copy.help_text = ""

        return func_copy


def _parse_matlab_function(code: str) -> dict:
    """Parse MATLAB function to extract name, inputs, outputs, and body.
    """
    lines = code.splitlines()
    function_info = {'name': '', 'inputs': [], 'outputs': [], 'body': [], 'help_text': []}

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('function'):
            # Parse function signature
            match = re.match(r'function\s+(?:\[?([^\]]*)\]?\s*=\s*)?(\w+)\s*\(([^)]*)\)', line)
            if match:
                outputs = match.group(1)
                name = match.group(2)
                inputs = match.group(3)

                function_info['name'] = name
                function_info['inputs'] = [i.strip() for i in inputs.split(',')] if inputs else []
                function_info['outputs'] = [o.strip() for o in outputs.split(',')] if outputs else []

                # Extract help text and body
                for j in range(i + 1, len(lines)):
                    line = lines[j].strip()
                    if line.startswith('%'):
                        function_info['help_text'].append(line[1:].strip())
                    elif line and not line.startswith('%'):
                        function_info['body'] = lines[j:]
                        break

                break

    return function_info
