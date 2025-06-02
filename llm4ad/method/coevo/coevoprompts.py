
from .coevoparas import CoEvoParas
from .prompts_str import *
from .prompts_str import init_prompts
from .prompts_str import off_prompts

class CoEvoPrompt:
    FORMAT_INTO_LIST_PROMPT = "List all the observations. Do not output anything else."
    FILTER_TO_USEFUL_LIST_PROMPT = "Filter these observations to the most useful."
    PARSE_INTO_PYTHON_LIST_PROMPT = \
    f"Format these observations as a Python list of strings. Below is an example:\n\n" \
    f"List of observations:\n" \
    f"1. Use XXX on YYY.\n" \
    f"2. ZZZ can be used on part A with transition B.\n\n" \
    f"Python output:\n" \
    f"```python\n[\n" \
    f"    \"Use XXX on YYY.\",\n" \
    f"    \"ZZZ can be used on part A with transition B.\"\n]\n```\n\n" \
    f"Output the Python output only."

    @classmethod
    def get_init_observation_prompt(
        cls, _task_description_str, coevo_paras: CoEvoParas, highest_level_obs: tuple[str, ...], obs_layer_i: int
    ):
        conv_content = init_prompts.get_observations_prompt(
            _task_description_str, highest_level_obs, coevo_paras.num_observations_to_generate, obs_layer_i
        )

        return conv_content

    @classmethod
    def get_init_impl_direct_prompt(
            cls, _task_description_str, coevo_paras: CoEvoParas, use_combo: tuple[str, ...]
    ):
        conv_content = init_prompts.get_direct_impl_prompts(
            _task_description_str, coevo_paras, use_combo
        )

        return conv_content

    @classmethod
    def get_offspring_observation_prompt(
            cls, task_info, coevo_paras: CoEvoParas,
            parents: list, mode: str = "crossover_positive",
            # use_summary: bool = False, summarizer=None
    ):
        prompt_content = off_prompts.get_observations_prompt(
            task_info, coevo_paras, parents, mode, coevo_paras.num_observations_to_generate
        )
        return prompt_content

    @classmethod
    def get_offspring_impl_indirect_prompt(cls
    ):
        prompt_content = off_prompts.get_impl_indirect()
        return prompt_content

    @classmethod
    def get_offspring_impl_direct_prompt(
            cls, coevo_paras: CoEvoParas,
            mode: str = "crossover_positive"
    ):
        prompt_content = off_prompts.get_impl_direct(mode, coevo_paras.rep_use)
        return prompt_content

    @classmethod
    def get_offspring_prompt(
            cls, task_info, evo_paras: CoEvoParas,
            parents: list, mode: str = "crossover_positive",
            use_summary: bool = False, summarizer=None
    ):
        if use_summary:
            assert summarizer is not None, "Summarizer is not provided."

        prompt_content = f"{task_info.task_info}\n# How to Respond\n"
        prompt_content += f"{get_offspring_how_to(mode)}"
        prompt_content += \
            f'First brainstorm ideas, then write down your thoughts process for solving the task ' \
            f'using the ideas, and finally provide the solution to the task in the required formats.'

        if use_summary:
            prompt_content += f' You will be provided with some effective ideas for inspiration, which may be helpful for you to solve the task.\n'
            idea_pool_search = summarizer.select_inspirations()
            prompt_content += \
                f'\nHere are some effective ideas which will help in solving the overall task:\n'
            prompt_content += f'{list_pool(idea_pool_search)}\n'

        prompt_content += \
            f'\nHere are {len(parents)} existing solutions with their ideas and evaluation results:\n' \
            f"{list_parents(evo_paras, parents)}\n"

        prompt_content += \
            '\nHints:\n' \
            f'- Ideas: Brainstorm at least {evo_paras.num_idea[0]} potential useful ideas for solving the task. ' \
            'Each idea should be innovative, creative, and non-obvious. Include the name, definition (brief description), and reasoning for each idea.\n' \
            f'- Thoughts: Think deeply step-by-step for solving the task using the ideas.\n' \
            f'- Solutions: Provide solution to the task in {len(evo_paras.rep_list)} formats, ' \
            f'[{evo_paras.rep_use.rep_name}] will be used for evaluation without any edit. Here are the formats:\n'
        for rep_i, rep in enumerate(evo_paras.rep_list):
            prompt_content += f'    [{rep.rep_name}]: {rep.rep_def}\n'

        prompt_content += f'{init_sol_response_format(evo_paras)}'
        return prompt_content

    @classmethod
    def get_single_summarizer_prompt(cls, task_info, evo_paras: CoEvoParas, previous_result_dict_list, inspiration_pool):


        prompt_content = f'{get_summarizer_how_to()}\n'

        prompt_content += f"Here is the task information:\n{task_info.task_info}\n"
        prompt_content += f'Here are the previous solutions:\n{list_single_sequential(evo_paras, previous_result_dict_list)}\n'
        prompt_content += \
            f'This are the ideas you’ve summarized:\n'\
            f'{list_pool(inspiration_pool)}\n'

        # Adding Summarization Instruction.
        prompt_content += f'{get_summarizer_end()}'
        return prompt_content

    @classmethod
    def get_offspring_summarizer_prompt(
            cls, task_info, evo_paras: CoEvoParas,
            parents_list, offspring_list,
            inspiration_pool
    ):
        prompt_content = f'{get_summarizer_how_to()}'

        prompt_content += f"Here is the task information:\n{task_info.task_info}\n"
        prompt_content += f'Here are the previous solutions:\n{list_parents(evo_paras, parents_list)}\n'
        prompt_content += f"{list_offsprings(evo_paras, offspring_list, len(parents_list))}"
        prompt_content += \
            f'This are the ideas you’ve summarized:\n' \
            f'{list_pool(inspiration_pool)}\n'

        # Adding Summarization Instruction.
        prompt_content += f'{get_summarizer_end()}'
        return prompt_content