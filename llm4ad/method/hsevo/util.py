from __future__ import annotations

import re
from typing import List, Dict


def format_messages(system: str, user: str) -> List[Dict[str, str]]:
    """Build a chat message list with a system and a user turn.

    This mirrors HSEvo's `format_messages`, but takes the system/user strings
    directly (the original took a Hydra `cfg` it never used). LLM4AD's
    `HttpsApi`/`OpenAIAPI` backends accept this list directly as the `prompt`
    argument of `draw_sample`, which preserves HSEvo's system/user separation.
    """
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_code_from_generator(content):
    """Extract code from the response of the code generator (verbatim HSEvo).

    The only deviation from upstream is that the prepended ``scipy``/``torch``
    imports are guarded with ``try/except`` so that environments without those
    optional packages (e.g. the default LLM4AD env, which ships without torch)
    do not fail to execute every generated heuristic.
    """
    pattern_code = r"```python(.*?)```"
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split("\n")
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("def"):
                start = i
            if "return" in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = "\n".join(lines[start : end + 1])

    if code_string is None:
        return None

    # --- LLM4AD adaptation -------------------------------------------------
    # Upstream HSEvo prepends:
    #   "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n"
    # We keep numpy/random/math unconditional, but guard scipy/torch so a
    # missing optional dependency degrades gracefully (only heuristics that
    # actually use the missing module fail, instead of the whole search).
    global_imports = (
        "import numpy as np\n"
        "import random\n"
        "import math\n"
        "try:\n"
        "    import scipy\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    pass\n"
    )
    code_string = global_imports + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements (verbatim HSEvo)."""
    lines = code_string.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith("def"):
            continue
        elif line.startswith("import"):
            continue
        elif line.startswith("from"):
            continue
        elif line.startswith("return"):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = "\n".join(filtered_lines)
    return code_string


def extract_to_hs(input_string: str):
    """Parse the harmony-search LLM response (verbatim HSEvo).

    Expects two ```python``` blocks: (1) the parameterised function with
    ``{param}`` placeholders in its default values and (2) a ``parameter_ranges``
    dict. Returns ``(parameter_ranges, function_block)`` or ``(None, None)``.
    """
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = (
            "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        )
        if any(
            keyword in parameter_ranges_block for keyword in ["inf", "np.inf", "None"]
        ):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals["parameter_ranges"]
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i : i + 3] == "def":
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if char == ":" and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[
            signature_start_index : signature_end_index + 1
        ]

        # Clean up the function signature from potential default values that might be corrupted (e.g. .eps suffix)
        # This regex looks for parameter definitions and cleans any trailing garbage before the next comma or closing paren
        function_signature = re.sub(
            r"(\w+\s*:\s*\w+\s*=\s*[\d.e-]+)(\.[a-zA-Z]+)", r"\1", function_signature
        )

        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(
                pattern, replacement, function_signature, flags=re.DOTALL
            )
        function_block = (
            function_block[:signature_start_index]
            + function_signature
            + function_block[signature_end_index + 1 :]
        )

    return parameter_ranges, function_block
