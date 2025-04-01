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
from collections.abc import Iterator, MutableSet
from typing import Sequence, Tuple, List, Dict, Any


class ModifyMatlabCode:
    @classmethod
    def add_global_statement(
            cls,
            program: str,
            variable_name: str) -> str:
        """Add global statement to the MATLAB function.
        Args:
            program      : The MATLAB program in string.
            variable_name: The name of the global variable.

        Example:
        ----------------------------------------------------------------------------------
        >>> program = '''
        ... function y = f(x)
        ...     y = x * 2;
        ... end'''
        >>> ModifyMatlabCode.add_global_statement(program, 'WEIGHT')
        'function y = f(x)\\n    global WEIGHT;\\n    y = x * 2;\\nend'
        """
        lines = program.split('\n')
        function_start = -1

        # Find the function declaration line
        for i, line in enumerate(lines):
            if line.strip().startswith('function'):
                function_start = i
                break

        if function_start >= 0:
            # Insert global statement after function declaration
            lines.insert(function_start + 1, f'    global {variable_name};')

        return '\n'.join(lines)

    @classmethod
    def add_path_statement(
            cls,
            program: str,
            path: str) -> str:
        """Add addpath statement to the beginning of the program.
        Args:
            program: The MATLAB program in string.
            path   : The path to add.
        """
        lines = program.split('\n')
        lines.insert(0, f"addpath('{path}');")
        return '\n'.join(lines)

    @classmethod
    def add_rng_seed(cls, program: str, seed: int = 2024) -> str:
        """Add rng seed statement to the MATLAB program.
        Args:
            program: program you want to add.
            seed   : seed number.
        """
        lines = program.split('\n')
        lines.insert(0, f'rng({seed});')
        return '\n'.join(lines)

    @classmethod
    def add_rng_seed_to_func(cls, program: str, func_name: str, seed: int = 2024) -> str:
        """Add rng seed statement to a specific MATLAB function.
        """
        lines = program.split('\n')
        in_target_function = False
        modified_lines = []

        for line in lines:
            if line.strip().startswith('function') and func_name in line:
                in_target_function = True
                modified_lines.append(line)
                modified_lines.append(f'    rng({seed});')
            else:
                modified_lines.append(line)

        return '\n'.join(modified_lines)

    @classmethod
    def replace_div_with_protected_div(
            cls,
            program: str,
            delta: float = 1e-5,
            return_div_func_name: bool = False
    ) -> str | Tuple[str, str]:
        """Replace division operations with protected division in MATLAB code."""
        protected_div_str = f'''
function y = protected_div(x, y)
    delta = {delta};
    y = x ./ (y + delta);
end
'''
        # Replace normal division with protected division
        modified_code = re.sub(r'([^\.])/', r'\1./', program)
        modified_code = re.sub(r'\./', r'protected_div(', modified_code)
        modified_code = modified_code + '\n' + protected_div_str

        if return_div_func_name:
            return modified_code, 'protected_div'
        return modified_code

    @classmethod
    def rename_function(cls, code: str, source_name: str, target_name: str) -> str:
        """Renames function calls from `source_name` to `target_name` in MATLAB code.
        """
        # Replace function definition
        pattern = f"function.*=.*{source_name}\\s*\\("
        replacement = lambda m: m.group(0).replace(source_name, target_name)
        code = re.sub(pattern, replacement, code)

        # Replace function calls
        pattern = f"\\b{source_name}\\s*\\("
        replacement = f"{target_name}("
        code = re.sub(pattern, replacement, code)

        return code

    @classmethod
    def get_functions_name(cls, code: str) -> MutableSet[str]:
        """Returns the set of all function names in MATLAB code.
        """
        function_names = set()
        lines = code.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('function'):
                # Extract function name from definition
                match = re.search(r'function\s+(?:\[?[^\]]*\]?\s*=\s*)?(\w+)\s*\(', line)
                if match:
                    function_names.add(match.group(1))

        return function_names

    @classmethod
    def add_help_text(cls, code: str, function_name: str, help_text: str) -> str:
        """Add help text (comments) to a MATLAB function.
        """
        lines = code.split('\n')
        in_function = False
        help_added = False
        modified_lines = []

        help_lines = [f'    % {line}' for line in help_text.split('\n')]

        for line in lines:
            if line.strip().startswith('function') and function_name in line:
                in_function = True
                modified_lines.append(line)
                modified_lines.extend(help_lines)
                help_added = True
            else:
                modified_lines.append(line)

        if not help_added:
            raise ValueError(f"Function {function_name} not found in code")

        return '\n'.join(modified_lines)


def _parse_matlab_function(code: str) -> Dict[str, Any]:
    """Parse MATLAB function definition to extract name, inputs, and outputs.
    """
    function_info = {}
    lines = code.split('\n')

    for line in lines:
        if line.strip().startswith('function'):
            # Match function definition pattern
            match = re.search(r'function\s+(?:\[?([^\]]*)\]?\s*=\s*)?(\w+)\s*\(([^)]*)\)', line)
            if match:
                outputs = match.group(1)
                name = match.group(2)
                inputs = match.group(3)

                function_info['name'] = name
                function_info['inputs'] = [i.strip() for i in inputs.split(',')] if inputs else []
                function_info['outputs'] = [o.strip() for o in outputs.split(',')] if outputs else []
                break

    return function_info


if __name__ == '__main__':
    program = '''
function y = f(x)
    y = x * 2;
end'''
    res = ModifyMatlabCode.add_global_statement(program, 'WEIGHT')
    print(res)
