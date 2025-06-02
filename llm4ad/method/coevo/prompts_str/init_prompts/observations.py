from coevo.tasks.base import TaskInfo

def system_prompt_observation(task_type: str, obs_layer: int):
    if task_type == "Python":
        identity = "Python programmer"
    elif task_type == "Verilog":
        identity = "Verilog engineer"
    else:
        raise NotImplementedError

    sys_prompt_obs = f"You are an expert {identity}. "

    if obs_layer == 0:
        sys_prompt_obs += \
            f'You will be given a problem and its details. ' \
            f"You will return several useful, non-obvious, and correct observations "\
            f"about the problem, like hints to solve the problem. "
    else:
        sys_prompt_obs += \
            f'You will be given a problem, its details and several observations about the problem.' \
            f"You will return several new, useful, and correct observations " \
            f"about the problem, derived from the given observations. "

    sys_prompt_obs += \
        "You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct."
    return sys_prompt_obs

def get_observations_prompt(
        task_info: TaskInfo, highest_level_obs: tuple[str, ...], num_observations_to_generate: int, obs_layer: int
) -> tuple[dict[str, str], dict[str, str]]:
    sys_prompt = system_prompt_observation(task_info.program_lang, obs_layer)
    if obs_layer == 0:
        user_prompt = get_observation(task_info, num_observations_to_generate)
    else:
        user_prompt = get_combine_observations(task_info, highest_level_obs, num_observations_to_generate)
    conversation = (
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    )
    return conversation


def get_observation(task_info: TaskInfo, num_observations: int) -> str:
    prompt = f"Here is the problem:\n\n{task_info.task_info}\n\n"
    prompt += \
            f"Brainstorm a list of {num_observations} innovative and non-obvious observations about " \
            f"properties of the problem. Each observation should be concise (a few words long), creatively insightful, "\
            f"and not immediately obvious or intuitively correct."
    prompt += \
            f"Before each observation, provide reasoning like this:\n\n" \
            f"Observation 1:\n" \
            f"[A paragraph or two of reasoning.]\n" \
            f"[Your observation] (e.g., 'XXX should be considered' or 'YYY can help with ZZZ.')\n\n" \
            f"Ensure observations are useful for solving the problem. Be concise and avoid restating the obvious."
    return prompt


def get_combine_observations(task_info: TaskInfo, obs_combo: tuple[str, ...], num_observations: int) -> str:
    prompt = f"Here is the problem:\n\n{task_info.task_info}\n\n"

    at_least_two = len(obs_combo) >= 2
    has_observation = len(obs_combo) >= 1

    if at_least_two:
        observation_str = "Here are several observations which may help in solving the problem:\n"
        observation_str += "- " + "\n- ".join(obs_combo) + "\n\n"
    elif has_observation:
        observation_str = "Here is a observation which may help in solving the problem:\n"
        observation_str += f"- {obs_combo[0]}\n\n"
    else:
        observation_str = "No observations are necessary to solve this problem.\n\n"

    prompt = prompt + observation_str
    prompt += \
    f"Start by reasoning about the implications of the {'observations' if has_observation else 'problem'}, " \
    f"include the most critical parts of the problem statement.\n" \
    f"Then creatively {'combine the observations' if at_least_two else 'use the implications'} "\
    f"to brainstorm at most {num_observations} non-obvious observations.\n"\
    f"Follow the rough format below:\n\n" \
    f"### Quotes:\n" \
    f"[Quotes from critical parts of the problem statement]\n" \
    f"{'[Quotes of the observations]'   if has_observation else ''}\n\n"\
    f"### Reasoned Implications:\n"\
    f"[Step-by-step reasoned-through implications{' of joining the observations above' if at_least_two else ('of the observation above' if has_observation else '')}, referencing the quotes above.]\n\n" \
    f"### New Observations:\n" \
    f"Observation 1:\n" \
    f"[Relevant quotes from the Reasoned Implications section.]\n" \
    f"[A paragraph or two of step-by-step reasoning.]\n" \
    f"[Your new observation]\n" \
    f"Observation 2:\n" \
    f"...\n\n"\
    f"Ensure new observations are creative, non-obvious, and useful for solving the problem. "\
    f"Ensure that the new observations are derived from the implications{' of the old observations' if has_observation else ''}. "\
    f"Be concise and avoid restating the obvious."
    return prompt