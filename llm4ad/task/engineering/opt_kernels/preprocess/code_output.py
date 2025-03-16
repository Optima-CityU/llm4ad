from abc import abstractmethod

from llm4ad.tools.llm.llm_api_https import HttpsApi

class CodeOutput(object):
    def __init__(self, llm: HttpsApi, parse_retry: int = 2, default_return = None):
        self.llm = llm
        self.parse_retry = parse_retry
        self.default_return = default_return

    @abstractmethod
    def get_code(self, code_content: str, previous_propose: str=None, previous_error: str=None) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _parse_response(self, response: str) -> str:
        raise NotImplementedError()

    def get_response_retry(self, prompt: list):
        parse_successful = False
        retry_count = 1

        while not parse_successful:
            try:
                response = self.llm.draw_sample(prompt)
                parsed_response = self._parse_response(response)
                return parsed_response
            except Exception as e:
                retry_count += 1
                if retry_count > self.parse_retry:
                    return self.default_return

