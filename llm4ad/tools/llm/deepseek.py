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


class DeepSeek(LLM):
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
                # params['llm_int8_enable_fp32_cpu_offload'] = True
            quantization_config = BitsAndBytesConfig(**params)
        else:
            torch_dtype = torch.float16

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path)

        # if isinstance(device_id, int):
        #     device_id = [device_id]
        # device_id = ','.join([str(i) for i in device_id])
        # os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        # print(f'os.environ["CUDA_VISIBLE_DEVICES"]={os.environ["CUDA_VISIBLE_DEVICES"]}')
        # print(self._tokenizer.eos_token_id)
        # print(self._tokenizer.pad_token_id)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            config=config,
            quantization_config=quantization_config,
            device_map={'model.embed_tokens': device_id, 'model.layers': device_id, 'model.norm': device_id, 'lm_head': device_id},
            torch_dtype=torch_dtype
            # cache_dir=None,
            # use_safetensors=False,
        )

    def draw_sample(self, prompt: Union[str, List[Dict[str, str]]], *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        inputs = self._tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt').to(self._model.device)

        output = self._model.generate(
            inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 8192),
            temperature=kwargs.get('temperature', 1.0),
            do_sample=kwargs.get('do_sample', True),
            top_k=kwargs.get('top_k', None),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1),
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id).cpu()

        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        output = self._tokenizer.decode(output[0, len(inputs[0]):], skip_special_tokens=True)
        return output
