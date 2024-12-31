from __future__ import annotations

import http.client
import json
from typing import Any
import traceback

from llm4ad.base import LLM


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=20, *, debug_mode=False):
        super().__init__()
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._debug_mode = debug_mode

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': kwargs.get('max_tokens', 4096),
                    'top_p': kwargs.get('top_p', None),
                    'temperature': kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                # print(data)
                response = data['choices'][0]['message']['content']
                return response
            except Exception as e:
                if self._debug_mode:
                    print(traceback.format_exc())
                    exit(0)
