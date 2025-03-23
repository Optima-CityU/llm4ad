from __future__ import annotations

import re
from typing import Tuple, List, Dict
from typing import Optional, Literal

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program


class EoHSampler:
    def __init__(self, sampler: LLM, template_program: str | Program):
        self._sampler = sampler
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str, code_type: Literal['Python', 'Kernel'] = 'Kernel') -> Tuple[str, Function]:
        response = self._sampler.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None
