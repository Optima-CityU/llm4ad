from __future__ import annotations

import concurrent.futures
import sys
import time
from functools import partial
from threading import Thread

from .config import EoHConfig
from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
# from .task_prompt import TaskPrompt
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)


class EoH:
    def __init__(self, task_description: str,
                 template_program: str | Program,
                 sampler: LLM,
                 evaluator: Evaluation,
                 profiler: EoHProfiler = None,
                 config: EoHConfig = EoHConfig(),
                 max_generations: int | None = 10,
                 max_sample_nums: int | None = None,
                 num_samplers: int = 4,
                 num_evaluators: int = 4,
                 *,
                 valid_only: bool = False,
                 resume_mode: bool = False,
                 initial_sample_num: int | None = None,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: str = 'thread',
                 **kwargs):
        """
        Args:
            task_description: a brief description of the algorithm design task and specification.
            template_program: the seed program (in str) as the initial function of the run.
                the template_program should be executable, i.e., incorporating package import, and function definition, and function body.
                The function body can be 'pass' in EoH as EoH does not request the template_program tobe valid.
            sampler         : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluator       : an instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            config          : an instance of 'llm4ad.method.eoh.config.EoHConfig'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums'.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations'.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        # arguments and keywords
        self._template_program_str = template_program
        self._task_description_str = task_description
        self._config = config
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._valid_only = valid_only
        self._debug_mode = debug_mode
        self._resume_mode = resume_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._config.pop_size)
        self._sampler = EoHSampler(sampler, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluator, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators

        # statistics
        self._tot_sample_nums = 0 if initial_sample_num is None else initial_sample_num

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_evaluators
            )

    def _sample_evaluate_register(self, prompt):
        """Sample a function using the given prompt -> evaluate it by submitting to the process/thread pool ->
        add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        # score
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        if self._profiler is not None:
            if self._valid_only:
                if score is not None:
                    self._profiler.register_function(func)
                    if isinstance(self._profiler, EoHProfiler):
                        self._profiler.register_population(self._population)
            else:
                self._profiler.register_function(func)
                if isinstance(self._profiler, EoHProfiler):
                    self._profiler.register_population(self._population)

        # update
        if self._valid_only:
            if score is not None:
                self._tot_sample_nums += 1
        else:
            self._tot_sample_nums += 1

        # append to the population
        self._population.register_function(func)

    def _thread_do_evolutionary_operator(self):
        def continue_loop():
            if self._max_generations is None and self._max_sample_nums is None:
                return True
            continue_until_reach_gen = False
            continue_until_reach_sample = False
            if self._max_generations is not None:
                if self._population.generation < self._max_generations:
                    continue_until_reach_gen = True
            if self._max_sample_nums is not None:
                if self._tot_sample_nums < self._max_sample_nums:
                    continue_until_reach_sample = True
            return continue_until_reach_gen and continue_until_reach_sample

        while continue_loop():
            try:
                # get a new func using e1
                indivs = [self._population.selection() for _ in range(self._config.selection_num)]
                prompt = EoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)

                if self._debug_mode:
                    print(prompt)
                    input()

                self._sample_evaluate_register(prompt)
                if not continue_loop():
                    break

                # get a new func using e2
                if self._config.use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._config.selection_num)]
                    prompt = EoHPrompt.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve)

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt)
                    if not continue_loop():
                        break

                # get a new func using m1
                if self._config.use_m1_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt)
                    if not continue_loop():
                        break

                # get a new func using m2
                if self._config.use_m2_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt)
                    if not continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    print(e)
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _thread_init_population(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                self._sample_evaluate_register(prompt)
            except Exception as e:
                if self._debug_mode:
                    print(e)
                continue

    def _init_population(self):
        # threads for sampling
        sampler_threads = [
            Thread(
                target=self._thread_init_population,
            ) for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def _do_sample(self):
        sampler_threads = [
            Thread(
                target=self._thread_do_evolutionary_operator,
            ) for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            # do init
            self._init_population()

        # do evolve
        self._do_sample()

        # finish
        if self._profiler is not None:
            self._profiler.finish()
