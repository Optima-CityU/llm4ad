from __future__ import annotations

import gc
import os
from typing import Dict, Union
from typing import List

import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

from llm4ad.base import LLM

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class Llama(LLM):
    def __init__(self,
                 pretrained_model_path,
                 device_id: int,
                 quantization=None):
        assert quantization in ['8b', '4b', None, ]
        super().__init__()

        quantization_config = None
        torch_dtype = None
        if quantization:
            params = {}
            if quantization == '4b':
                params['load_in_4bit'] = True
                params['bnb_4bit_compute_dtype'] = torch.float16
            if quantization == '8b':
                params['load_in_8bit'] = True
                params['llm_int8_enable_fp32_cpu_offload'] = True
            quantization_config = BitsAndBytesConfig(**params)
        else:
            torch_dtype = torch.float16

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            config=config,
            quantization_config=quantization_config,
            # device_map='auto',
            torch_dtype=torch_dtype
            # cache_dir=None,
            # use_safetensors=False,
        ).to(f'cuda:{device_id}')

    def _format_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(prompt, str):
            sys_prompt = ''
            usr_prompt = prompt
        else:
            if prompt[0]['role'] == 'system':
                sys_prompt = prompt[0]['content']
                usr_prompt = prompt[1]['content']
            else:
                sys_prompt = ''
                usr_prompt = prompt[0]['content']
        prompt = f'''<s>[INST] <<SYS>>
{sys_prompt}
<</SYS>>
{usr_prompt} [/INST]'''
        return prompt

    def draw_sample(self, prompt: Union[str, List[Dict[str, str]]], *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        prompt = self._format_prompt(prompt)
        inputs = self._tokenizer.encode(
            prompt,
            max_length=kwargs.get('max_length'),
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self._model.device)

        output = self._model.generate(
            inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 8192),
            temperature=kwargs.get('temperature', 1.0),
            do_sample=kwargs.get('do_sample', True),
            top_k=kwargs.get('top_k', None),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1),
            eos_token_id=2,
            pad_token_id=2).cpu()

        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        output = self._tokenizer.decode(output[0, len(inputs[0]):], skip_special_tokens=True)
        return output
