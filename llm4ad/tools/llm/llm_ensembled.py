import argparse
import threading
from typing import List, Any

from ...base import LLM

class EnsembledLLM(LLM):
    def __init__(self, llms: List[LLM]):
        super().__init__()
        self._llms = llms
        self._count = 0
        self._count_lock = threading.Lock()

    def _get_count(self):
        try:
            self._count_lock.acquire()
            cur_count = self._count
            self._count += 1
            if self._count == len(self._llms):
                self._count = 0
            return cur_count
        finally:
            self._count_lock.release()

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        count = self._get_count()
        return self._llms[count].draw_sample(prompt, *args, **kwargs)
