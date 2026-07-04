from __future__ import annotations

import copy

from ...base import Function

# ----------------------------------------------------------------------------
# Common prompt templates (verbatim copies of prompts/common/*.txt)
# ----------------------------------------------------------------------------

# system_generator.txt
SYSTEM_GENERATOR = """{seed} Your task is to design heuristics that can effectively solve optimization problems.
Your response outputs Python code and nothing else. Format your code as a Python code string: "```python ... ```".
"""

# user_generator.txt
USER_GENERATOR = """{seed} Your task is to write a {func_name} function for {problem_desc}
{func_desc}
"""

# seed.txt
SEED = """{seed_func}

Refer to the format of a trivial design above. Be very creative and give `{func_name}_v2`. Output code only and enclose your code with Python code block: ```python ... ```."""

# crossover.txt
CROSSOVER = """{user_generator}

### Better code
{func_signature_m1}
{code_method1}

### Worse code
{func_signature_m2}
{code_method2}

### Analyze & experience
- {analyze}
- {exp}

Your task is to write an improved function `{func_name}_v2` by COMBINING elements of two above heuristics base Analyze & experience.
Output the code within a Python code block: ```python ... ```, has comment and docstring (<50 words) to description key idea of heuristics design.

I'm going to tip $999K for a better heuristics! Let's think step by step."""

# mutation.txt
MUTATION = """{user_generator}

Current heuristics:
{func_signature1}
{elitist_code}

Now, think outside the box write a mutated function `{func_name}_v2` better than current version.
You can use some hints below:
- {reflection}

Output code only and enclose your code with Python code block: ```python ... ```.
I'm going to tip $999K for a better solution!"""

# system_reflector.txt
SYSTEM_REFLECTOR = """You are an expert in the domain of optimization heuristics. Your task is to provide useful advice based on analysis to design better heuristics.
"""

# user_flash_reflection.txt
USER_FLASH_REFLECTION = """### List heuristics
Below is a list of design heuristics ranked from best to worst.
{lst_method}

### Guide
- Keep in mind, list of design heuristics ranked from best to worst. Meaning the first function in the list is the best and the last function in the list is the worst.
- The response in Markdown style and nothing else has the following structure:
"**Analysis:**
**Experience:**"
In there:
+ Meticulously analyze comments, docstrings and source code of several pairs (Better code - Worse code) in List heuristics to fill values for **Analysis:**.
Example: "Comparing (best) vs (worst), we see ...;  (second best) vs (second worst) ...; Comparing (1st) vs (2nd), we see ...; (3rd) vs (4th) ...; Comparing (second worst) vs (worst), we see ...; Overall:"

+ Self-reflect to extract useful experience for design better heuristics and fill to **Experience:** (<60 words).

I'm going to tip $999K for a better heuristics! Let's think step by step."""

# user_comprehensive_reflection.txt
USER_COMPREHENSIVE_REFLECTION = """Your task is to redefine 'Current self-reflection' paying attention to avoid all things in 'Ineffective self-reflection' in order to come up with ideas to design better heuristics.

### Current self-reflection
{curr_reflection}
{good_reflection}

### Ineffective self-reflection
{bad_reflection}

Response (<100 words) should have 4 bullet points: Keywords, Advice, Avoid, Explanation.
I'm going to tip $999K for a better heuristics! Let's think step by step."""

# system_harmony_search.txt
SYSTEM_HARMONY_SEARCH = """You are an expert in code review. Your task extract all threshold, weight or hardcode variable of the function make it become default parameters."""

# harmony_search.txt
HARMONY_SEARCH = """[code]
{code_extract}

Now extract all threshold, weight or hardcode variable of the function make it become default parameters and give me a 'parameter_ranges' dictionary representation. Key of dict is name of variable. Value of key is a tuple in Python MUST include 2 float elements, first element is begin value, second element is end value corresponding with parameter.

- Output code only and enclose your code with Python code block: ```python ... ```.
- Output 'parameter_ranges' dictionary only and enclose your code with other Python code block: ```python ... ```."""


# ----------------------------------------------------------------------------
# Scientist personas (verbatim from hsevo.py) used to diversify the initial
# population: each initial individual is generated with a rotating persona.
# ----------------------------------------------------------------------------
SCIENTISTS = [
    "You are an expert in the domain of optimization heuristics.",
    "You are Albert Einstein, relativity theory developer.",
    "You are Isaac Newton, the father of physics.",
    "You are Marie Curie, pioneer in radioactivity.",
    "You are Nikola Tesla, master of electricity.",
    "You are Galileo Galilei, champion of heliocentrism.",
    "You are Stephen Hawking, black hole theorist.",
    "You are Richard Feynman, quantum mechanics genius.",
    "You are Rosalind Franklin, DNA structure revealer.",
    "You are Ada Lovelace, computer programming pioneer.",
]


# ----------------------------------------------------------------------------
# Problem-specific prompt derivation (replaces HSEvo's per-problem .txt files).
# These build the equivalents of seed_func.txt / func_signature.txt /
# func_desc.txt from the LLM4AD task's template function + task description.
# ----------------------------------------------------------------------------


def make_func_signature(function: Function) -> str:
    """Return a signature template with a ``{version}`` placeholder, e.g.
    ``def priority_v{version}(item: float, bins: np.ndarray) -> np.ndarray:``.
    """
    return_type = f" -> {function.return_type}" if function.return_type else ""
    return f"def {function.name}_v{{version}}({function.args}){return_type}:"


def make_seed_func(function: Function) -> str:
    """Render the template function as ``{name}_v1`` inside a python code block.

    HSEvo's seed_func.txt holds a complete trivial design. We render the LLM4AD
    template function (with its docstring + body) as ``_v1`` and wrap it in a
    ```python``` fence so that HSEvo's `extract_code_from_generator` extracts the
    full body via its regex path (rather than the def->first-return fallback).
    """
    f = copy.deepcopy(function)
    f.name = f"{function.name}_v1"
    return "```python\n" + str(f).rstrip() + "\n```"


def make_func_desc(function: Function, task_description: str) -> str:
    """Describe the target function's I/O. Prefer the template docstring; fall
    back to the task description."""
    if function.docstring:
        return function.docstring
    return task_description or ""
