from __future__ import annotations

import re
from typing import Tuple, List, Dict
from typing import Optional, Literal

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program
from ...base.opt_kernels.sample import CPPSampleTrimmer


class EoHSampler:
    def __init__(self, sampler: LLM, template_program: str | Program, code_type: Literal['Python', 'Kernel'] = 'Kernel'):
        self._sampler = sampler
        self._template_program = template_program
        self.code_type = code_type

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self._sampler.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        if self.code_type == 'Python':
            code = SampleTrimmer.trim_preface_of_function(response)
            function = SampleTrimmer.sample_to_function(code, self._template_program)
        else:
            code = CPPSampleTrimmer.trim_preface_of_function(response)
            function = CPPSampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None
