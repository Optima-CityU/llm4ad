from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import math
import time
import traceback
from threading import Lock
from typing import Literal, Optional

import numpy as np

from .profiler import HSEvoProfiler
from .prompt import (
    SYSTEM_GENERATOR,
    USER_GENERATOR,
    SEED,
    CROSSOVER,
    MUTATION,
    SYSTEM_REFLECTOR,
    USER_FLASH_REFLECTION,
    USER_COMPREHENSIVE_REFLECTION,
    SYSTEM_HARMONY_SEARCH,
    HARMONY_SEARCH,
    SCIENTISTS,
    make_func_signature,
    make_seed_func,
    make_func_desc,
)
from .util import (
    extract_code_from_generator,
    filter_code,
    extract_to_hs,
    format_messages,
)
from ...base import (
    Evaluation,
    LLM,
    Function,
    Program,
    TextFunctionProgramConverter,
    SecureEvaluator,
)
from ...tools.profiler import ProfilerBase


class HSEvo:
    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase = None,
        max_sample_nums: Optional[int] = 450,
        pop_size: Optional[int] = 10,
        init_pop_size: Optional[int] = 30,
        mutation_rate: float = 0.5,
        hm_size: int = 5,
        hmcr: float = 0.7,
        par: float = 0.5,
        bandwidth: float = 0.2,
        max_iter: int = 5,
        num_samplers: int = 4,
        num_evaluators: int = 4,
        *,
        resume_mode: bool = False,
        debug_mode: bool = False,
        multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
        **kwargs,
    ):
        """HSEvo: Diversity-Driven Harmony Search + Genetic Algorithm using LLMs.

        Args:
            llm             : an instance of 'llm4ad.base.LLM'.
            evaluation      : an instance of 'llm4ad.base.Evaluation'.
            profiler        : an instance of 'llm4ad.method.hsevo.HSEvoProfiler'. Pass 'None' to disable.
            max_sample_nums : terminate after this many evaluated functions (maps to HSEvo's 'max_fe').
            pop_size        : population size used for selection/crossover.
            init_pop_size   : number of individuals generated for the initial population (with rotating personas).
            mutation_rate   : fraction of pop_size mutated from the elitist each generation.
            hm_size         : harmony-memory size for the harmony search operator.
            hmcr            : harmony memory considering rate.
            par             : pitch adjustment rate.
            bandwidth       : pitch adjustment bandwidth (fraction of each parameter range).
            max_iter        : number of harmony-search improvisation iterations.
            num_samplers    : number of threads used for batched LLM sampling.
            num_evaluators  : number of workers used for parallel evaluation.
            resume_mode     : if True, skip the initial population creation (see note in `run`).
            debug_mode      : if True, print detailed information.
            multi_thread_or_process_eval: 'thread' or 'process' pool for evaluation.
            **kwargs        : extra args passed to 'llm4ad.base.SecureEvaluator' (e.g. 'fork_proc').
        """
        # ----- LLM4AD framework handles -----
        self._llm = llm
        self._profiler = profiler
        self._max_sample_nums = max_sample_nums
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators

        # ----- HSEvo hyper-parameters (names mirror the original cfg.*) -----
        self.mutation_rate = mutation_rate
        self.init_pop_size = init_pop_size
        self.pop_size = pop_size
        self.hm_size = hm_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        # Additive temperature boost applied ONLY when sampling the initial
        # population (mirrors upstream's `cfg.temperature + 0.3`). Applied via
        # `_temporarily_raise_temperature` if the LLM exposes `_temperature`.
        self.init_temperature_boost = float(kwargs.pop("init_temperature_boost", 0.3))

        # ----- HSEvo state (verbatim) -----
        self.iteration = 0
        self.function_evals = 0
        # generation counter + operator success bookkeeping (logging/verification only)
        self._generation = 0
        self._op_stats = {"crossover": 0, "mutation": 0, "harmony_search": 0}
        self._op_ok_gens = {
            "crossover": set(),
            "mutation": set(),
            "harmony_search": set(),
        }
        self.elitist = None
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.long_term_reflection_str = ""
        self.lst_good_reflection = []
        self.lst_bad_reflection = []
        self.population = []
        self.seed_ind = None
        self.str_flash_memory = {"analyze": "", "exp": ""}
        # LLM4AD maximizes `score`; HSEvo minimizes `obj`. We always map obj = -score.
        self.obj_type = "min"

        # ----- Problem definition derived from the LLM4AD task -----
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_str
        )
        self._function_to_evolve: Function = (
            TextFunctionProgramConverter.text_to_function(self._template_program_str)
        )
        if self._function_to_evolve is None or self._template_program is None:
            raise ValueError(
                "HSEvo: could not parse the task `template_program` into a single function."
            )
        self.func_name = self._function_to_evolve.name
        self.problem_desc = self._task_description_str
        self.func_signature = make_func_signature(self._function_to_evolve)
        self.seed_func = make_seed_func(self._function_to_evolve)
        self.func_desc = make_func_desc(
            self._function_to_evolve, self._task_description_str
        )
        self.external_knowledge = ""
        self.str_comprehensive_memory = self.external_knowledge

        # ----- Prompts (verbatim HSEvo common templates) -----
        self.system_generator_prompt = SYSTEM_GENERATOR
        self.user_generator_prompt = USER_GENERATOR
        self.crossover_prompt = CROSSOVER
        self.mutation_prompt = MUTATION
        self.system_reflector_prompt = SYSTEM_REFLECTOR
        self.user_flash_reflection_prompt = USER_FLASH_REFLECTION
        self.user_comprehensive_reflection_prompt = USER_COMPREHENSIVE_REFLECTION
        self.system_hs_prompt = SYSTEM_HARMONY_SEARCH
        self.hs_prompt = HARMONY_SEARCH
        self.seed_prompt = SEED.format(
            seed_func=self.seed_func, func_name=self.func_name
        )
        self.scientists = SCIENTISTS

        # ----- print-once flags (verbatim) -----
        self.print_crossover_prompt = True
        self.print_mutate_prompt = True
        self.print_flash_reflection_prompt = True
        self.print_comprehensive_reflection_prompt = True
        self.print_hs_prompt = True
        self.local_sel_hs = None

        # ----- evaluator + executors -----
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # ----- sampling bookkeeping (for profiler sample_time / operator) -----
        self._cur_sample_time = 0.0
        self._cur_operator = "init"
        self._uid_counter = 0
        self._uid_lock = Lock()

        logging.info("Problem: " + str(self.func_name))
        logging.info("Function name: " + str(self.func_name))

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    # ------------------------------------------------------------------
    # Boundary 1: LLM adapter (replaces HSEvo's multi_chat_completion)
    # ------------------------------------------------------------------
    def _safe_draw(self, messages) -> str:
        """Draw one sample. Pass the [system, user] message list directly to the
        backend (HttpsApi/OpenAIAPI support it). Fall back to a concatenated
        single string for backends that only accept strings."""
        try:
            return self._llm.draw_sample(messages)
        except Exception:
            try:
                text = "\n\n".join(
                    m.get("content", "") for m in messages if isinstance(m, dict)
                )
                return self._llm.draw_sample(text)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                return ""

    def _draw_batch(self, messages_list, n: int = 1):
        """Mimic HSEvo's `multi_chat_completion(messages_list, n, model, temperature)`.

        - If a single message list is given, it is wrapped into a batch.
        - `n > 1` duplicates the single prompt `n` times (LLM4AD backends do not
          expose an `n` argument), matching HSEvo's multi-sample-per-prompt use.
        - The per-call temperature/model used by upstream are ignored (the LLM
          instance is pre-configured).
        """
        if len(messages_list) > 0 and not isinstance(messages_list[0], list):
            messages_list = [messages_list]
        tasks = messages_list if n == 1 else messages_list * n
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(self._num_samplers, 1)
        ) as ex:
            responses = list(ex.map(self._safe_draw, tasks))
        elapsed = time.time() - start
        self._cur_sample_time = elapsed / max(len(tasks), 1)
        return responses

    @contextlib.contextmanager
    def _temporarily_raise_temperature(self, delta: float):
        """Temporarily add `delta` to the LLM's sampling temperature.

        LLM4AD's `LLM.draw_sample` has no per-call temperature argument, so this
        mutates a `_temperature` attribute for backends that expose one (e.g.
        the `VLLMChat` client used in the HSEvo comparison, which reads
        `self._temperature` on every call). It is a no-op for backends without
        such an attribute or when `delta == 0`. The original value is always
        restored, even on error. Used to reproduce upstream's +0.3 init boost.
        """
        llm = self._llm
        if not delta or not hasattr(llm, "_temperature"):
            yield
            return
        original = llm._temperature
        try:
            llm._temperature = original + delta
            logging.info(
                f"Init sampling temperature raised {original} -> {llm._temperature} "
                f"(+{delta}) for initial-population diversity."
            )
            yield
        finally:
            llm._temperature = original

    def _next_uid(self) -> str:
        with self._uid_lock:
            self._uid_counter += 1
            return f"ind_{self._uid_counter}"

    # ------------------------------------------------------------------
    # Boundary 2: evaluation adapter (replaces HSEvo's subprocess eval)
    # ------------------------------------------------------------------
    def _code_to_program(self, code: str) -> Optional[Program]:
        """Convert a generated code string into an LLM4AD `Program`.

        The generated function's full signature (including any default values
        introduced by harmony search) is kept; only the function NAME is
        normalized to the task's canonical name so name-based lookups work.
        """
        try:
            gen_program = TextFunctionProgramConverter.text_to_program(code)
            if gen_program is None or len(gen_program.functions) != 1:
                # LLM4AD's SecureEvaluator assumes exactly one top-level function.
                return None
            gen_func = gen_program.functions[0]
            gen_func.name = self.func_name
            return Program(preface=gen_program.preface, functions=[gen_func])
        except Exception:
            return None

    def _register_function(
        self, program: Program, score, sample_time, evaluate_time, operator
    ):
        if self._profiler is None:
            return
        func = TextFunctionProgramConverter.program_to_function(program)
        if func is None:
            return
        func.score = score
        func.sample_time = sample_time
        func.evaluate_time = evaluate_time
        func.operator = operator
        if func.docstring:
            func.algorithm = func.docstring
        self._profiler.register_function(func, program=str(program))

    def evaluate_population(
        self, population: list[dict], hs_try_idx: int = None
    ) -> list[dict]:
        """Evaluate a population using LLM4AD's SecureEvaluator (parallel).

        Mirrors HSEvo's `evaluate_population` accounting (one `function_evals`
        increment per individual, best-obj logging), but the inner evaluation is
        delegated to `SecureEvaluator` and `obj = -score`.
        """
        programs = []
        futures = []

        # submit
        for response_id in range(len(population)):
            self.function_evals += 1
            individual = population[response_id]

            program = None
            if individual["code"] is not None:
                program = self._code_to_program(individual["code"])
            programs.append(program)

            if program is None:
                self.mark_invalid_individual(
                    individual, "Invalid response / unparseable code!"
                )
                futures.append(None)
                continue

            try:
                futures.append(
                    self._evaluation_executor.submit(
                        self._evaluator.evaluate_program_record_time, program
                    )
                )
            except Exception as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                self.mark_invalid_individual(individual, str(e))
                futures.append(None)

        # collect
        for response_id, future in enumerate(futures):
            individual = population[response_id]
            program = programs[response_id]
            if future is None:
                continue
            try:
                score, eval_time = future.result()
            except Exception as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                score, eval_time = None, None

            if score is None or (isinstance(score, float) and math.isnan(score)):
                self.mark_invalid_individual(individual, "Invalid objective value!")
            else:
                individual["exec_success"] = True
                individual["obj"] = -float(
                    score
                )  # LLM4AD maximizes score => HSEvo minimizes obj

            self._register_function(
                program,
                score,
                individual.get("_sample_time"),
                eval_time,
                individual.get("_operator", "unknown"),
            )

        # Log after all population is evaluated
        valid_objs = [ind["obj"] for ind in population if ind.get("exec_success")]
        best_obj = min(valid_objs) if valid_objs else float("inf")
        logging.info(f"Eval={self.function_evals}, BestObj={best_obj}")

        return population

    # ------------------------------------------------------------------
    # Individual bookkeeping (verbatim, minus on-disk artifacts)
    # ------------------------------------------------------------------
    def response_to_individual(
        self, response: str, response_id: int, file_name: str = None
    ) -> dict:
        """Convert an LLM response (or a substituted code string) to an individual."""
        code = extract_code_from_generator(response)
        individual = {
            "code_path": self._next_uid(),
            "code": code,
            "response_id": response_id,
            "tryHS": False,
            "_sample_time": self._cur_sample_time,
            "_operator": self._cur_operator,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """Mark an individual as invalid (verbatim)."""
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    # ------------------------------------------------------------------
    # Initialization (verbatim, LLM/eval boundaries swapped)
    # ------------------------------------------------------------------
    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        self._cur_operator = "init"
        self._cur_sample_time = 0.0
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + str(code))
        seed_ind = {
            "code_path": self._next_uid(),
            "code": code,
            "response_id": 0,
            "tryHS": False,
            "_sample_time": 0.0,
            "_operator": "init",
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(
                "Seed function is invalid. Please check the task template_program."
            )

        self.update_iter()

        messages_lst = []
        for i in range(self.init_pop_size):
            user_generator_prompt_full = self.user_generator_prompt.format(
                seed=self.scientists[i % len(self.scientists)],
                func_name=self.func_name,
                problem_desc=self.problem_desc,
                func_desc=self.func_desc,
            )
            system_generator_prompt_full = self.system_generator_prompt.format(
                seed=self.scientists[i % len(self.scientists)]
            )
            system = system_generator_prompt_full
            user = (
                user_generator_prompt_full
                + "\n"
                + self.seed_prompt
                + "\n"
                + self.long_term_reflection_str
            )
            messages = format_messages(system, user)
            messages_lst.append(messages)

        # Upstream HSEvo raises the sampling temperature by +0.3 for the initial
        # population to diversify it (main.py: `self.cfg.temperature + 0.3`).
        # LLM4AD's `LLM.draw_sample` has no per-call temperature, but backends
        # that expose a mutable `_temperature` (e.g. the VLLMChat used for the
        # HSEvo comparison) can be bumped for the duration of the init batch and
        # restored afterwards. This restores upstream's init diversity, which is
        # what prevents the population from collapsing onto the best-fit
        # attractor (see _compare/ analysis).
        self._cur_operator = "init"
        with self._temporarily_raise_temperature(self.init_temperature_boost):
            responses = self._draw_batch(messages_lst, 1)
        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    # ------------------------------------------------------------------
    # Update / selection (verbatim)
    # ------------------------------------------------------------------
    def update_iter(self) -> None:
        """Update after each iteration (verbatim)."""
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        self.iteration += 1

    def random_select(self, population: list[dict]) -> list[dict]:
        """Random selection with equal probability (verbatim).

        Note: HSEvo's black-box branch (which also filters `obj < seed_obj`) is
        omitted because LLM4AD has no notion of `problem_type`; we always keep
        the valid-individuals branch.
        """
        selected_population = []
        # Eliminate invalid individuals
        population = [
            individual for individual in population if individual["exec_success"]
        ]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical;
            # otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    # ------------------------------------------------------------------
    # Reflection (verbatim, LLM boundary swapped)
    # ------------------------------------------------------------------
    def flash_reflection(self, population: list[dict]) -> None:
        self._cur_operator = "flash_reflection"
        lst_str_method = []
        seen_elements = set()

        sorted_population = sorted(population, key=lambda x: x["obj"], reverse=False)
        for idx, individual in enumerate(sorted_population):
            suffix = (
                "th"
                if 11 <= idx + 1 <= 13
                else {1: "st", 2: "nd", 3: "rd"}.get((idx + 1) % 10, "th")
            )
            str_idx_method = f"[Heuristics {idx + 1}{suffix}]"
            str_code = individual["code"]
            temp_str = str_idx_method + "\n" + str_code + "\n"

            if temp_str not in seen_elements:
                seen_elements.add(temp_str)
                lst_str_method.append(temp_str)

        system = self.system_reflector_prompt
        user = self.user_flash_reflection_prompt.format(
            problem_desc=self.problem_desc,
            lst_method="\n".join(lst_str_method),
            schema_reflection={"analyze": "str", "exp": "str"},
        )
        messages = format_messages(system, user)

        if self.print_flash_reflection_prompt:
            logging.info(
                "Flash reflection Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_flash_reflection_prompt = False

        flash_reflection_res = self._draw_batch([messages], 1)[0]
        analyze_start = flash_reflection_res.find("**Analysis:**") + len(
            "**Analysis:**"
        )
        exp_start = flash_reflection_res.find("**Experience:**")

        analysis_text = flash_reflection_res[analyze_start:exp_start].strip()
        experience_text = flash_reflection_res[
            exp_start + len("**Experience:**") :
        ].strip()

        # Create the JSON structure
        self.str_flash_memory = {"analyze": analysis_text, "exp": experience_text}

    def comprehensive_reflection(self):
        self._cur_operator = "comprehensive_reflection"
        system = self.system_reflector_prompt

        good_reflection = (
            "\n\n".join(self.lst_good_reflection)
            if len(self.lst_good_reflection) > 0
            else "None"
        )
        bad_reflection = (
            "\n\n".join(self.lst_bad_reflection)
            if len(self.lst_bad_reflection) > 0
            else "None"
        )

        user = self.user_comprehensive_reflection_prompt.format(
            bad_reflection=bad_reflection,
            good_reflection=good_reflection,
            curr_reflection=self.str_flash_memory["exp"],
        )
        messages = format_messages(system, user)

        if self.print_comprehensive_reflection_prompt:
            logging.info(
                "Comprehensive reflection Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_comprehensive_reflection_prompt = False

        comprehensive_response = self._draw_batch([messages], 1)[0]
        self.str_comprehensive_memory = (
            self.external_knowledge + "\n" + comprehensive_response
        )

    # ------------------------------------------------------------------
    # Crossover / mutation (verbatim, LLM boundary swapped)
    # ------------------------------------------------------------------
    def crossover(self, population: list[dict]) -> list[dict]:
        self._cur_operator = "crossover"
        messages_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            if population[i]["obj"] < population[i + 1]["obj"]:
                parent_1 = population[i]
                parent_2 = population[i + 1]
            else:
                parent_1 = population[i + 1]
                parent_2 = population[i]

            # Crossover
            system = self.system_generator_prompt.format(seed=self.scientists[0])
            func_signature_m1 = self.func_signature.format(version=0)
            func_signature_m2 = self.func_signature.format(version=1)
            user_generator_prompt_full = self.user_generator_prompt.format(
                seed=self.scientists[0],
                func_name=self.func_name,
                problem_desc=self.problem_desc,
                func_desc=self.func_desc,
            )
            user = self.crossover_prompt.format(
                user_generator=user_generator_prompt_full,
                func_signature_m1=func_signature_m1,
                func_signature_m2=func_signature_m2,
                code_method1=filter_code(parent_1["code"]),
                code_method2=filter_code(parent_2["code"]),
                analyze=self.str_flash_memory["analyze"],
                exp=self.str_comprehensive_memory,
                func_name=self.func_name,
            )
            messages = format_messages(system, user)
            messages_lst.append(messages)

            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info(
                    "Crossover Prompt: \nSystem Prompt: \n"
                    + system
                    + "\nUser Prompt: \n"
                    + user
                )
                self.print_crossover_prompt = False

        # Asynchronously generate responses
        response_lst = self._draw_batch(messages_lst, 1)
        crossed_population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(response_lst)
        ]

        assert len(crossed_population) == self.pop_size
        return crossed_population

    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        self._cur_operator = "mutation"
        system = self.system_generator_prompt.format(seed=self.scientists[0])
        func_signature1 = self.func_signature.format(version=1)
        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=self.scientists[0],
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )

        user = self.mutation_prompt.format(
            user_generator=user_generator_prompt_full,
            reflection=self.str_comprehensive_memory,
            func_signature1=func_signature1,
            elitist_code=filter_code(self.elitist["code"]),
            func_name=self.func_name,
        )
        messages = format_messages(system, user)

        if self.print_mutate_prompt:
            logging.info(
                "Mutation Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_mutate_prompt = False

        responses = self._draw_batch(
            [messages], int(self.pop_size * self.mutation_rate)
        )
        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]
        return population

    # ------------------------------------------------------------------
    # Harmony search (verbatim, LLM/eval boundaries swapped)
    # ------------------------------------------------------------------
    def sel_individual_hs(self):
        candidate_hs = [
            individual for individual in self.population if individual["tryHS"] is False
        ]
        best_candidate_id = self.find_best_obj(candidate_hs)
        self.local_sel_hs = best_candidate_id
        # NOTE: preserved verbatim from upstream HSEvo (best_candidate_id is an
        # index into `candidate_hs` but is used to index `self.population`).
        self.population[best_candidate_id]["tryHS"] = True
        return self.population[best_candidate_id]["code"]

    def initialize_harmony_memory(self, bounds):
        problem_size = len(bounds)
        harmony_memory = np.zeros((self.hm_size, problem_size))
        for i in range(problem_size):
            lower_bound, upper_bound = bounds[i]
            harmony_memory[:, i] = np.random.uniform(
                lower_bound, upper_bound, self.hm_size
            )
        return harmony_memory

    def responses_to_population(self, responses, try_hs_idx=None) -> list[dict]:
        """Convert responses (here, substituted code strings) to a population."""
        population = []
        for response_id, response in enumerate(responses):
            individual = self.response_to_individual(response, response_id)
            population.append(individual)
        return population

    def create_population_hs(
        self, str_code, parameter_ranges, harmony_memory, try_hs_idx=None
    ):
        str_create_pop = []
        for i in range(len(harmony_memory)):
            tmp_str = str_code
            for j in range(len(list(parameter_ranges))):
                tmp_str = tmp_str.replace(
                    ("{" + list(parameter_ranges)[j] + "}"), str(harmony_memory[i][j])
                )
                if tmp_str == str_code:
                    return None
            str_create_pop.append("```python\n" + tmp_str + "\n```")

        population_hs = self.responses_to_population(str_create_pop, try_hs_idx)
        return self.evaluate_population(population_hs, try_hs_idx)

    def find_best_obj(self, population_hs):
        objs = [individual["obj"] for individual in population_hs]
        best_solution_id = np.argmin(np.array(objs))
        return best_solution_id

    def create_new_harmony(self, harmony_memory, bounds):
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for i in range(harmony_memory.shape[1]):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = harmony_memory[
                    np.random.randint(0, harmony_memory.shape[0]), i
                ]
                if np.random.rand() < self.par:
                    adjustment = (
                        np.random.uniform(-1, 1)
                        * (bounds[i][1] - bounds[i][0])
                        * self.bandwidth
                    )
                    new_harmony[i] += adjustment
            else:
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return new_harmony

    def update_harmony_memory(
        self,
        population_hs,
        harmony_memory,
        new_harmony,
        func_block,
        parameter_ranges,
        try_hs_idx,
    ):
        objs = [individual["obj"] for individual in population_hs]
        worst_index = np.argmax(np.array(objs))

        new_individual = self.create_population_hs(
            func_block, parameter_ranges, [new_harmony.tolist()], try_hs_idx
        )[0]

        if new_individual["obj"] < population_hs[worst_index]["obj"]:
            population_hs[worst_index] = new_individual
            harmony_memory[worst_index] = new_harmony
        return population_hs, harmony_memory

    def harmony_search(self):
        # Safety guard (LLM4AD addition): if there is no candidate left to tune,
        # skip rather than crashing on an empty argmin.
        if not any(individual["tryHS"] is False for individual in self.population):
            return None

        self._cur_operator = "harmony_search"
        system = self.system_hs_prompt
        user = self.hs_prompt.format(code_extract=self.sel_individual_hs())
        messages = format_messages(system, user)
        # Print get hs prompt for the first iteration
        if self.print_hs_prompt:
            logging.info(
                "Harmony Search Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_hs_prompt = False

        responses = self._draw_batch([messages], 1)

        logging.info("LLM Response for HS step: " + str(responses[0]))
        parameter_ranges, func_block = extract_to_hs(responses[0])
        if parameter_ranges is None or func_block is None:
            return None
        bounds = [value for value in parameter_ranges.values()]

        harmony_memory = self.initialize_harmony_memory(bounds)
        population_hs = self.create_population_hs(
            func_block, parameter_ranges, harmony_memory
        )

        if population_hs is None:
            return None
        elif (
            len(
                [
                    individual
                    for individual in population_hs
                    if individual["exec_success"] is True
                ]
            )
            == 0
        ):
            self.function_evals -= self.hm_size
            return None

        # [HS-CHECK]
        init_objs = [ind["obj"] for ind in population_hs if ind["exec_success"]]
        n_distinct = len(set(init_objs))
        init_best = min(init_objs) if init_objs else float("inf")

        for iteration in range(self.max_iter):
            new_harmony = self.create_new_harmony(harmony_memory, bounds)
            population_hs, harmony_memory = self.update_harmony_memory(
                population_hs,
                harmony_memory,
                new_harmony,
                func_block,
                parameter_ranges,
                iteration,
            )
        best_obj_id = self.find_best_obj(population_hs)
        population_hs[best_obj_id]["tryHS"] = True
        hs_best = population_hs[best_obj_id]["obj"]
        logging.info(
            f"[HS-CHECK] iter={self.iteration} hm_size={self.hm_size} "
            f"valid={len(init_objs)} distinct_init_objs={n_distinct} "
            f"init_best={init_best} hs_best={hs_best} "
            f"improved_over_init={hs_best < init_best}"
        )
        return population_hs[best_obj_id]

    # ------------------------------------------------------------------
    # Evolutionary loop (verbatim, minus on-disk artifacts)
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_count(population: list[dict]) -> int:
        return sum(1 for ind in population if ind.get("exec_success"))

    def evolve(self):
        while self.function_evals < self._max_sample_nums:
            self._generation += 1
            gen = self._generation
            logging.info(
                f"===== [Gen {gen}] start (function_evals={self.function_evals}, "
                f"best_obj={self.best_obj_overall}) ====="
            )
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(
                    "All individuals are invalid. Please check the task evaluation."
                )
            # Select
            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )  # add elitist to population for selection
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            logging.info(
                f"[Gen {gen}] selection: OK ({len(selected_population)} parents)"
            )

            # Reflection
            self.flash_reflection(selected_population)
            flash_ok = bool(self.str_flash_memory.get("analyze")) or bool(
                self.str_flash_memory.get("exp")
            )
            logging.info(
                f"[Gen {gen}] flash_reflection: {'OK' if flash_ok else 'EMPTY'} "
                f"(analyze={len(self.str_flash_memory.get('analyze', ''))} chars, "
                f"exp={len(self.str_flash_memory.get('exp', ''))} chars)"
            )
            self.comprehensive_reflection()
            comp_ok = bool(
                self.str_comprehensive_memory and self.str_comprehensive_memory.strip()
            )
            logging.info(
                f"[Gen {gen}] comprehensive_reflection: {'OK' if comp_ok else 'EMPTY'} "
                f"({len(self.str_comprehensive_memory)} chars)"
            )
            curr_code_path = self.elitist["code_path"]

            # Crossover
            crossed_population = self.crossover(selected_population)
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            n_cross_valid = self._valid_count(crossed_population)
            logging.info(
                f"[Gen {gen}] crossover: {n_cross_valid}/{len(crossed_population)} valid"
            )
            self._op_stats["crossover"] += n_cross_valid
            if n_cross_valid > 0:
                self._op_ok_gens["crossover"].add(gen)
            # Update
            self.update_iter()

            # Mutate
            mutated_population = self.mutate()
            # Evaluate
            evaluated_mut = self.evaluate_population(mutated_population)
            self.population.extend(evaluated_mut)
            n_mut_valid = self._valid_count(mutated_population)
            logging.info(
                f"[Gen {gen}] mutation: {n_mut_valid}/{len(mutated_population)} valid"
            )
            self._op_stats["mutation"] += n_mut_valid
            if n_mut_valid > 0:
                self._op_ok_gens["mutation"].add(gen)
            # Update
            self.update_iter()

            if curr_code_path != self.elitist["code_path"]:
                self.lst_good_reflection.append(self.str_flash_memory["exp"])
            else:
                self.lst_bad_reflection.append(self.str_flash_memory["exp"])

            # Harmony Search
            try_hs_num = 3
            individual_hs = None
            while try_hs_num:
                individual_hs = self.harmony_search()
                if individual_hs is not None:
                    self.population.extend([individual_hs])
                    break
                else:
                    try_hs_num -= 1
            if individual_hs is not None:
                logging.info(
                    f"[Gen {gen}] harmony_search: OK (obj={individual_hs.get('obj')})"
                )
                self._op_stats["harmony_search"] += 1
                self._op_ok_gens["harmony_search"].add(gen)
            else:
                logging.info(f"[Gen {gen}] harmony_search: FAILED after 3 tries")
            self.update_iter()

            logging.info(
                f"===== [Gen {gen}] done (function_evals={self.function_evals}, "
                f"best_obj={self.best_obj_overall}) ====="
            )

            # Optional population checkpoint
            if self._profiler is not None and isinstance(self._profiler, HSEvoProfiler):
                self._profiler.register_population(self.population, self.iteration)

        return self.best_code_overall, self.best_code_path_overall

    # ------------------------------------------------------------------
    # LLM4AD entry point
    # ------------------------------------------------------------------
    def run(self):
        if not self._resume_mode:
            # do initialization (upstream HSEvo runs this inside __init__)
            self.init_population()

        # evolutionary search
        try:
            self.evolve()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.info(f"HSEvo evolution terminated: {e}")
            if self._debug_mode:
                traceback.print_exc()
        finally:
            # finish
            if self._profiler is not None:
                self._profiler.finish()
            self._llm.close()
            try:
                self._evaluation_executor.shutdown(cancel_futures=True)
            except Exception:
                pass
