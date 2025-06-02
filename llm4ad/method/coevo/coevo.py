# Module Name: CoEvo
# Last Revision: 2025/6/2
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:

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

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .population import Population
from .profiler import EoHProfiler
from .prompt import CoEvoPrompt
from .prompt_cpp import CoEvoPromptCPP
from .sampler import EoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...base.opt_kernels import KERFunction,KERProgram, KERTextFunctionProgramConverter
from ...base.opt_kernels.evaluate import CPPSecureEvaluator
from ...tools.profiler import ProfilerBase

import numpy as np
from .coevo_node import ObservationNode
from .coevoprompts import CoEvoPrompt
from .coevoparas import CoEvoParas


class CoEvo:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 code_type: Literal['Python', 'Kernel'] = 'Kernel',
                 resume_mode: bool = False,
                 initial_sample_nums_max: int = 50,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            initial_sample_nums_max     : maximum samples restriction during initialization.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = initial_sample_nums_max
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self.code_type = code_type
        if code_type == 'Python':
            self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
            self._function_to_evolve_name: str = self._function_to_evolve.name
            self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        elif code_type == 'Kernel':
            self._function_to_evolve = KERTextFunctionProgramConverter.text_to_function(evaluation.cuda_code)
            self._py_func_ref = KERTextFunctionProgramConverter.text_to_function_py(evaluation.func_code)
            self._function_to_evolve_name: str = self._function_to_evolve.name
            self._template_program = KERTextFunctionProgramConverter.text_to_program(evaluation.cuda_code)


            # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = EoHSampler(llm, self._template_program_str, code_type=code_type)
        if code_type == 'Python':
            self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        elif code_type == 'Kernel':
            self._evaluator = CPPSecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = max(
            self._initial_sample_nums_max,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

        self.coevo_paras = CoEvoParas()
        self.llm_inst = llm
        self.init_tree_list = []

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        # convert to Program instance
        if self.code_type == 'Python':
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        elif self.code_type == 'Kernel':
            program = KERTextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        # register to profiler
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

        # register to the population
        self._population.register_function(func)

    def _gen_obs_tree(self):
        obs_node = ObservationNode(self._task_description_str, self.coevo_paras, 0, ())
        for layer_i in range(self.coevo_paras.num_obs_layers):
            highest_nodes = obs_node.get_highest_level_nodes()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_task = [executor.submit(every_high_node.get_observation, self.llm_inst) for every_high_node in
                            highest_nodes]
                concurrent.futures.wait(all_task)
        return obs_node

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_eoh_operator(self):
        while self._continue_loop():
            try:
                all_combos = []
                for each_tree in self.init_tree_list:
                    highest_level_combo = each_tree.get_highest_level_combos()
                    non_empty_highest_level_combo = []
                    for every_combo in highest_level_combo:
                        if len(every_combo) >= 1:
                            non_empty_highest_level_combo.append(every_combo)
                    combo_idx = np.random.permutation(len(non_empty_highest_level_combo))
                    use_combos = [non_empty_highest_level_combo[idx] for idx in combo_idx]
                    all_combos.extend(use_combos)
                combo_idx = np.random.permutation(len(all_combos))
                all_combos = [all_combos[idx] for idx in combo_idx][:3]
                modified_task_str = self._task_description_str + f'Here are some useful observations that may help:\n\n'
                for each_combo in all_combos:
                    modified_task_str += f' - {each_combo}\n'
                # get a new func using e1
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                if self.code_type=='Python':
                    prompt = CoEvoPrompt.get_prompt_e1(modified_task_str, indivs, self._function_to_evolve)
                elif self.code_type=='Kernel':
                    prompt = CoEvoPromptCPP.get_prompt_e1(modified_task_str, indivs, self._function_to_evolve)
                if self._debug_mode:
                    print(f'E1 Prompt: {prompt}')
                self._sample_evaluate_register(prompt)
                if not self._continue_loop():
                    break

                # get a new func using e2
                if self._use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._selection_num)]
                    if self.code_type == 'Python':
                        prompt = CoEvoPrompt.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve)
                    elif self.code_type == 'Kernel':
                        prompt = CoEvoPromptCPP.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                # get a new func using m1
                if self._use_m1_operator:
                    indiv = self._population.selection()
                    if self.code_type == 'Python':
                        prompt = CoEvoPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
                    elif self.code_type == 'Kernel':
                        prompt = CoEvoPromptCPP.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                # get a new func using m2
                if self._use_m2_operator:
                    indiv = self._population.selection()
                    if self.code_type == 'Python':
                        prompt = CoEvoPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
                    elif self.code_type == 'Kernel':
                        prompt = CoEvoPromptCPP.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            # do initialization
            pop_size = 0
            while self._population.generation == 0:
                new_obs_tree = self._gen_obs_tree()
                self.init_tree_list.append(new_obs_tree)
                highest_level_combo = new_obs_tree.get_highest_level_combos()
                non_empty_highest_level_combo = []
                for every_combo in highest_level_combo:
                    if len(every_combo) >= 1:
                        non_empty_highest_level_combo.append(every_combo)
                non_empty_highest_level_combo.append(())
                combo_idx = np.random.permutation(len(non_empty_highest_level_combo))[:self.coevo_paras.num_sol_from_a_tree]
                use_combos = [non_empty_highest_level_combo[idx] for idx in combo_idx]

                use_combos = use_combos[:self._pop_size-pop_size]

                all_impl_direct_prompts = [CoEvoPrompt.get_init_impl_direct_prompt(self._task_description_str, self.coevo_paras, use_combo) for use_combo in use_combos]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    all_task = [executor.submit(self._sample_evaluate_register, impl_prompt) for impl_prompt in all_impl_direct_prompts]
                    concurrent.futures.wait(all_task)
                all_results_direct = [task.result() for task in all_task]
                pop_size += len(use_combos)
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    break
            # terminate searching if
            if len(self._population) < self._selection_num:
                print(f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                      f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                      f'Please also check your evaluation implementation and LLM implementation.')
                return
        # evolutionary search
        self._multi_threaded_sampling(self._iteratively_use_eoh_operator)
        # finish
        if self._profiler is not None:
            self._profiler.finish()
