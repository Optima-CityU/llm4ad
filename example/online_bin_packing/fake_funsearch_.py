from typing import Any

import openai
import wandb

from llm4ad.base import Evaluation, LLM


class MyEvaluator(Evaluation):
    def __init__(self):
        super().__init__(timeout_seconds=10)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        return callable_func()


evaluator = MyEvaluator()


# class MySampler(Sampler):
#     def __init__(self):
#         super().__init__()
#
#     def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
#         my_prompt = 'asdfasdfadsf' + prompt
#




class OpenAIAPI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=30, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt,
            stream=False,
        )
        return response.choices[0].message.content


sampler = OpenAIAPI(base_url='https://api.bltcy.ai/v1',
                    api_key='sk-z4tSPqfaQ74KBpTlFc4f7fC767D04603Be0696F7A8BcC7D8', model='gpt-4o-mini')


template_program = '''
import numpy as np

def fake_func() -> int:
    """Returns a integer."""
    return 0
'''


from llm4ad.method.funsearch import FunSearch
from llm4ad.method.funsearch import FunSearchWandbProfiler

wandb.login(key='579d2d16d85fdeabe12fac0f5eccf903a2c594da')

# profiler = FunSearchWandbProfiler(
#     log_dir=f'test_log', wandb_project_name='huqinglong', name='run1'
# )
#
#
# funsearch = FunSearch(
#     template_program=template_program,
#     evaluator=evaluator,
#     sampler=sampler,
#     profiler=profiler,
#     max_sample_nums=20,
# )
#
# funsearch.run()



from llm4ad.method.eoh import EoH, EoHConfig, EoHWandbProfiler
config = EoHConfig(pop_size=5)

profiler = EoHWandbProfiler(log_dir='test_log_eoh', wandb_project_name='huqinglong', name='eoh_run1')

eoh = EoH(
    task_description='这个任务要求你返回一个尽可能大的整数，当然这个整数要小于100',
    template_program=template_program,
    evaluator=evaluator,
    sampler=sampler,
    profiler=profiler,
    max_sample_nums=20,
    config=config,
)

eoh.run()
