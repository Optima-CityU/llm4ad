from __future__ import annotations

import concurrent.futures
import copy
import time
from threading import Thread

from ...base import *
from .config import RegEvoPopulationConfig
from .population import RegEvoPopulation
from .profiler import RegEvoProfiler, ProfilerBase


class RegEvoSearch:
    def __init__(
            self,
            template_program: str,
            sampler: LLM,
            evaluator: Evaluation,
            profiler: ProfilerBase = None,
            reevo_pop_config: RegEvoPopulationConfig = RegEvoPopulationConfig(),
            max_sample_nums: int | None = 20,
            valid_only: bool = False,
            functions_per_prompt: int = 2,
            samples_per_prompt: int = 4,
            num_samplers: int = 4,
            num_evaluators: int = 4,
            pop_size: int = 100,
            *,
            initial_sample_num: int | None = None,
            debug_mode: bool = False,
            multi_thread_or_process_eval: str = 'thread',
            resume_mode: bool = False
    ):
        # arguments and keywords
        self._template_program_str = template_program
        self._max_sample_nums = max_sample_nums
        self._debug_mode = debug_mode
        self._resume_mode = resume_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # population, sampler, and evaluator
        self._population = RegEvoPopulation(pop_size, reevo_pop_config)
        self._sampler = SampleTrimmer(sampler)
        self._sampler_trimmer = SampleTrimmer(sampler)
        self._evaluator = SecureEvaluator(evaluator, debug_mode=debug_mode)
        self._profiler = profiler
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._functions_per_prompt = functions_per_prompt
        self._samples_per_prompt = samples_per_prompt
        self._valid_only = valid_only

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

        # threads for sampling
        self._sampler_threads = [
            Thread(target=self._sample_evaluate_register) for _ in range(self._num_samplers)
        ]

    def _sample_evaluate_register(self):
        while (self._max_sample_nums is None) or (self._tot_sample_nums < self._max_sample_nums):
            try:
                # get prompt
                funcs_per_prompt = min(len(self._population), self._functions_per_prompt)
                prompt_contents = ([self._get_prompt(num=funcs_per_prompt)] * self._samples_per_prompt)

                # do sample
                draw_sample_start = time.time()
                sampled_funcs = self._sampler.draw_samples(prompt_contents)
                draw_sample_times = time.time() - draw_sample_start
                avg_time_for_each_sample = draw_sample_times / len(sampled_funcs)

                # convert samples to program instances
                programs_to_be_eval = []
                for func in sampled_funcs:
                    program = SampleTrimmer.sample_to_program(func, self._template_program)
                    # if sample to program success
                    if program is not None:
                        programs_to_be_eval.append(program)

                # submit tasks to the thread pool and evaluate
                futures = []
                for program in programs_to_be_eval:
                    future = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program)
                    futures.append(future)
                # get evaluate scores and evaluate times
                scores_times = [f.result() for f in futures]
                scores, times = [i[0] for i in scores_times], [i[1] for i in scores_times]

                # register to program database and profiler
                for program, score, eval_time in zip(programs_to_be_eval, scores, times):
                    # update
                    self._tot_sample_nums += 1
                    # convert to Function instance
                    function = TextFunctionProgramConverter.program_to_function(program)
                    # check if the function has converted to Function instance successfully
                    if function is None:
                        continue
                    # register to program database
                    if score is not None:
                        function.score = score
                        self._population.register_function(function)
                    # register to profiler
                    if self._profiler is not None:
                        function.score = score
                        function.sample_time = avg_time_for_each_sample
                        function.evaluate_time = eval_time
                        self._profiler.register_function(function)
                        # self._profiler.register_program_db(self._database)

                    if self._valid_only:
                        if score is not None:
                            self._profiler.register_function(function)
                            if isinstance(self._profiler, RegEvoProfiler):
                                self._profiler.register_regevo_pop(self._population)
                    else:
                        self._profiler.register_function(function)
                        if isinstance(self._profiler, RegEvoProfiler):
                            self._profiler.register_regevo_pop(self._population)
            except Exception as e:
                # raise e
                continue

    def _get_prompt(self, num: int):
        # selection
        funcs = self._population.selection(num)
        funcs = copy.deepcopy(funcs)

        # remove docstrings
        for i in range(len(funcs)):
            assert isinstance(funcs[i], Function) and funcs[i].score is not None
            funcs[i] = SampleTrimmer.remove_docstrings(funcs[i])

        # sort by scores
        funcs = sorted(funcs, key=lambda f: f.score)
        func_name = self._function_to_evolve_name

        content = ''
        for v, func in enumerate(funcs):
            if v == 0:
                prog = TextFunctionProgramConverter.function_to_program(func, self._template_program)
                prog.functions[0].name = f'{func_name}_v0'
                content += str(prog)
            else:
                func.name = f'{func_name}_v{v}'
                func.docstring = f'Improved version of \'{func_name}_v{v - 1}'
                content += str(func)

        template_func = TextFunctionProgramConverter.program_to_function(self._template_program)
        template_func.name = f'{func_name}_v{len(funcs)}'
        template_func.docstring = f"Improved version of '{func_name}_v{len(funcs) - 1}'"
        template_func.body = ''
        content += str(template_func)
        return content

    def run(self):
        if not self._resume_mode:
            # evaluate the template program, make sure the score of which is not 'None'
            score, eval_time = self._evaluator.evaluate_program_record_time(program=self._template_program)
            if score is None:
                raise RuntimeError('The score of the template function must not be "None".')

            # register the template program to the program database
            self._function_to_evolve.score = score
            self._population.register_function(self._function_to_evolve)
            if self._profiler:
                self._function_to_evolve.score = score
                self._function_to_evolve.evaluate_time = eval_time
                self._profiler.register_function(self._function_to_evolve)

        # start sampling using multiple threads
        for t in self._sampler_threads:
            t.start()

        # join all threads to the main thread
        for t in self._sampler_threads:
            t.join()

        if self._profiler is not None:
            self._profiler.finish()
