import re

from llm4ad.tools.llm.llm_api_https import HttpsApi
from .prompts import CudaTranslatePrompts
from ..code_output import CodeOutput


class CudaCodeOutput(CodeOutput):
    def __init__(self, llm: HttpsApi):
        super().__init__(llm)

    def get_code(self, code_content: str, previous_propose: str=None, previous_error: str=None) -> str:
        sys_prompt = CudaTranslatePrompts.SYS_PROMPT
        user_prompt = CudaTranslatePrompts.get_translate_prompt(code_content)
        prompt = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        if previous_propose is not None:
            content_new = CudaTranslatePrompts.get_error_prompt(previous_error)
            prompt.append({'role': 'assistant', 'content': f"<cuda>\n\n```c++\n\n{previous_propose}\n\n```\n\n</cuda>"})
            prompt.append({'role': 'user', 'content': content_new})

        parsed_response = self.get_response_retry(prompt)

        return parsed_response

    def _parse_response(self, response: str) -> str:
        # match the content between <cuda> and </cuda> tags
        code_pattern = re.compile(r'<cuda>(.*?)</cuda>', re.DOTALL)

        # find the first match
        match = code_pattern.search(response)
        if match is None:
            return response
        else:
            code = match.group(1)
            code_block_pattern = re.compile(
                r'^\s*```([^\n]*)\n([\s\S]*?)\n```\s*$',
                re.DOTALL
            )
            cleaned_code = code_block_pattern.sub(r'\2', code.strip())
            return cleaned_code.strip()
