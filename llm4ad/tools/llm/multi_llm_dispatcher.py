from __future__ import annotations

import threading
from queue import Queue
from typing import List, Dict, Union

from llm4ad.base import LLM


class MultiEngineDispatcher(LLM):
    def __init__(self, engines: List[LLM]):
        super().__init__()
        self._engines = engines
        self._lock = threading.Lock()
        self._idle_queue = Queue()
        self._condition = threading.Condition(self._lock)

        # init their state to idle
        for i in range(len(self._engines)):
            self._idle_queue.put(i)

    def draw_sample(self, prompt: Union[str, List[Dict[str, str]]], *args, **kwargs) -> str:
        model_index = self._get_idle_model()
        model = self._engines[model_index]
        response = model.draw_sample(prompt)
        self._mark_model_as_idle(model_index)
        return response

    def _get_idle_model(self):
        with self._condition:
            while self._idle_queue.empty():
                self._condition.wait()
            return self._idle_queue.get()

    def _mark_model_as_idle(self, model_index):
        with self._condition:
            self._idle_queue.put(model_index)
            self._condition.notify()
