# Module Name: MLES
# Last Revision: 2026/2/9
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu, Xialiang Tong, Mingxuan Yuan, Fei Liu, Zhichao Lu, and Qingfu Zhang.
#       "Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies."
#       The Fourteenth International Conference on Learning Representations (ICLR). 2026.

# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from threading import Lock
from typing import List, Dict, Optional

from .population import Population
from ...base import Function
from ...tools.profiler import TensorboardProfiler, ProfilerBase, WandBProfiler


class MLESProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 evaluation_name='Problem',
                 method_name='EoH',
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 run_mode='Training',
                 **kwargs):
        """MLES Profiler
        Args:
            log_dir            : the directory of current run
            evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
            method_name        : the name of the search method.
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        self.run_mode = run_mode
        if self.run_mode == 'Using':
            create_random_path = False
            using_algo_designed_path = kwargs.get('using_algo_designed_path', '')
            if using_algo_designed_path == '':
                raise ValueError(f"You are using 'Using' mode but give no designed algorithm")
            else:
                log_dir = os.path.join(log_dir, using_algo_designed_path)

        super().__init__(evaluation_name=evaluation_name,
                         method_name=method_name,
                         log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._cur_gen = 0
        self._pop_lock = Lock()
        if self.run_mode == 'Training' or self.run_mode == 'Combined':
            if self._log_dir:
                self._ckpt_dir = os.path.join(self._log_dir, 'population')
                os.makedirs(self._ckpt_dir, exist_ok=True)
                self._output_dir = os.path.join(self._log_dir, 'designed_result')
                os.makedirs(self._output_dir, exist_ok=True)
        if self.run_mode == 'Using' or self.run_mode == 'Combined':
            if self._log_dir:
                self._using_dir = os.path.join(self._log_dir, 'using', self._result_folder + '_U')
                os.makedirs(self._using_dir, exist_ok=True)

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self._num_samples == 0 or
                    pop.generation == self._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_json = {
                    'algorithm': f.algorithm,
                    'function': str(f),
                    'score': f.score
                }

                if hasattr(f, 'operator'):
                    f_json['operator'] = f.operator

                if hasattr(f, 'pop_register_number'):
                    f_json['pop_register_number'] = f.pop_register_number

                if hasattr(f, 'parents'):
                    f_json['parents'] = f.parents

                if hasattr(f, 'image64'):
                    f_json['image64'] = f.image64

                if hasattr(f, 'response'):
                    f_json['response'] = f.response

                if hasattr(f, 'prompt'):
                    f_json['prompt'] = f.prompt

                if hasattr(f, 'observation'):
                    f_json['observation'] = f.observation

                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file, indent=4)
            self._cur_gen += 1

        except Exception as e:
            print(f"Error occurred when register the population: {e}")

        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _write_json(self, function: Function, program='', *, record_type='history', record_sep=200):
        """Write function data to a JSON file.
        Args:
            function   : The function object containing score and string representation.
            record_type: Type of record, 'history' or 'best'. Defaults to 'history'.
            record_sep : Separator for history records. Defaults to 200.
        """
        assert record_type in ['history', 'best']

        if not self._log_dir:
            return

        generation_num = self._cur_gen

        sample_order = self._num_samples
        content = {
            'sample_order': sample_order,
            'generation': generation_num,
            'score': function.score,
            'operator': function.operator,
            'algorithm': function.algorithm,  # Added when recording
            'function': str(function),
        }

        if hasattr(function, 'pop_register_number'):
            content['pop_register_number'] = function.pop_register_number

        if hasattr(function, 'parents'):
            content['parents'] = function.parents

        if hasattr(function, 'response'):
            content['response'] = function.response

        if hasattr(function, 'prompt'):
            content['prompt'] = function.prompt

        if hasattr(function, 'observation'):
            content['observation'] = function.observation

        if record_type == 'history':
            lower_bound = (sample_order // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


    def using_final(self, **kwargs):
        final_results = kwargs.get('final_results')
        output_file_path = os.path.join(self._using_dir, 'using_final_output.json')
        with open(output_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=4)


class EoHTensorboardProfiler(TensorboardProfiler, MLESProfiler):

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """EoH Profiler for Tensorboard.
        Args:
            log_dir            : the directory of current run
            evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        MLESProfiler.__init__(
            self, log_dir=log_dir,
            create_random_path=create_random_path,
            **kwargs
        )
        TensorboardProfiler.__init__(
            self,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )

    def finish(self):
        if self._log_dir:
            self._writer.close()


class EoHWandbProfiler(WandBProfiler, MLESProfiler):
    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """EoH Profiler for Wandb.
        Args:
            wandb_project_name : the name of the wandb project
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        MLESProfiler.__init__(
            self,
            log_dir=log_dir,
            create_random_path=create_random_path,
            **kwargs
        )
        WandBProfiler.__init__(
            self,
            wandb_project_name=wandb_project_name,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def finish(self):
        self._wandb.finish()
