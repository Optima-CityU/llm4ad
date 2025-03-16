import re

from .prompts import FuncConvertPrompts
from ..code_output import CodeOutput

from llm4ad.tools.llm.llm_api_https import HttpsApi


class FuncCodeOutput(CodeOutput):
    def __init__(self, llm: HttpsApi):
        super().__init__(llm)

    def get_code(self, code_content: str, previous_propose: str=None, previous_error: str=None) -> str:
        sys_prompt = FuncConvertPrompts.SYS_PROMPT
        user_prompt = FuncConvertPrompts.get_conversion_prompt(code_content)

        prompt = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        if previous_propose is not None:
            content_new = FuncConvertPrompts.get_error_prompt(previous_error)
            prompt.append({'role': 'assistant', 'content': f"```python\n{previous_propose}\n```"})
            prompt.append({'role': 'user', 'content': content_new})

        parsed_response = self.get_response_retry(prompt)

        return parsed_response

    def _parse_response(self, response: str) -> str:
        code_block_pattern = re.compile(
            r'^\s*```(?:\w+)?\s*\n?(.*?)\n?\s*```\s*$',
            re.DOTALL
        )
        cleaned_code = code_block_pattern.sub(r'\1', response.strip())
        return cleaned_code.strip()
