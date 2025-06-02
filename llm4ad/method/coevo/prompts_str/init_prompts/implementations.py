from coevo.methods.coevo.reps import BaseRep
from coevo.tasks.base import TaskInfo
from coevo.methods.coevo.coevoparas import CoEvoParas

def get_indirect_impl_prompts(
        task_info: TaskInfo, coevo_paras: CoEvoParas, use_combo: tuple[str, ...]
):
    sys_prompt = system_prompt_impl_indirect(task_info.program_lang, coevo_paras.rep_list, coevo_paras.rep_use)
    user_prompt = get_impl_indirect(task_info, use_combo)
    conversation = (
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    )
    return conversation

def get_direct_impl_prompts(
        task_info: TaskInfo, coevo_paras: CoEvoParas, indirect_sol: str
):
    sys_prompt = system_prompt_impl_direct(task_info.program_lang, coevo_paras.rep_list, coevo_paras.rep_use)
    user_prompt = get_impl_direct(task_info, indirect_sol, coevo_paras.rep_use)
    conversation = (
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    )
    return conversation

def system_prompt_impl_direct(
        task_type: str, rep_list: list[BaseRep] , rep_use: BaseRep
):
    if task_type == "Python":
        identity = "Python programmer"
    elif task_type == "Verilog":
        identity = "Verilog engineer"
    else:
        raise NotImplementedError

    sys_prompt_obs = \
        f"You are an expert {identity}. " \
        f"You will be given a problem, its details and:"
    for rep_i, each_rep in enumerate(rep_list):
        sys_prompt_obs += f'{rep_i+1}. {each_rep.rep_def}\n'

    sys_prompt_obs += \
        f"\nYou will return {rep_use.rep_def} and passes all tests. "\
        f"You will NOT return anything except for {rep_use.rep_def} inside markdown codeblocks."

    return sys_prompt_obs

def system_prompt_impl_indirect(
        task_type: str, rep_list: list[BaseRep], rep_use: BaseRep
):
    if task_type == "Python":
        identity = "Python programmer"
    elif task_type == "Verilog":
        identity = "Verilog engineer"
    else:
        raise NotImplementedError

    sys_prompt_obs = \
        f"You are an expert {identity}. " \
        f'You will be given a problem, its details and several observations necessary to solve the problem.' \
        f" You will return:\n"
    for rep_i, each_rep in enumerate(rep_list):
        sys_prompt_obs += f'{rep_i+1}. {each_rep.rep_def}\n'

    sys_prompt_obs += \
        f"\nYou will NOT return any {rep_use.rep_def}. Be as creative as possible, "\
        f"going beyond what you think is intuitively correct."

    return sys_prompt_obs


def get_impl_direct(task_info: TaskInfo,  indirect_sol: str, rep_use: BaseRep) -> str:
    prompt = f"Here is the problem:\n\n{task_info.task_info}\n\n"
    prompt += f"Natural language tutorial:\n\n{indirect_sol}\n\n"
    prompt += f"Please write {rep_use.rep_def} inside markdown codeblocks."
    return prompt

def get_impl_indirect(task_info: TaskInfo,  use_combo: tuple[str, ...]) -> str:
    prompt = f"Here is the problem:\n\n{task_info.task_info}\n\n"

    if len(use_combo) == 0:
        prompt += "No observations are necessary to solve this problem.\n\n"
    else:
        prompt += "Here are the intelligent observations to help solve the problem:\n\n"
        prompt += " - " + "\n\n - ".join(use_combo) + "\n\n"

    prompt += \
        "Use the observations above to brainstorm a natural language solution to the problem above.\n" \
        "1. Go beyond intuition and generate simple, innovative ideas.\n" \
        "2. Before each step, quote EXACTLY the relevant part of the observations. This is crucial.\n"
    return prompt