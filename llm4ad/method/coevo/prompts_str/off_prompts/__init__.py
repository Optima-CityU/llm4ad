from coevo.tasks.base import TaskInfo
from coevo.methods.coevo.reps import BaseRep
from coevo.methods.coevo.coevoparas import CoEvoParas
def system_prompt_observation(task_type: str, mode: str, num_parents: int) -> str:
    if task_type == "Python":
        identity = "Python programmer"
    elif task_type == "Verilog":
        identity = "Verilog engineer"
    else:
        raise NotImplementedError

    sys_prompt_obs = \
        f"You are an expert {identity}. "\
        f'You will be given a problem, its details and {num_parents} existing solutions with their ideas and evaluation results. ' \
        f"You will return several useful, non-obvious, and correct observations for "

    prompt_content = {
                "crossover_positive": 'creating a new solution that has a totally different form from the given solutions but can be motivated from the existing ones. ',
                "crossover_negative": 'creating a new solution that has a totally different form from the given solutions. ',
                "mutation_positive": 'creating a solution in different forms but can be a modified version of the existing solution. ',
                "mutation_negative": 'creating a totally different solution from the existing solution. '
            }[mode]

    return sys_prompt_obs + prompt_content

def get_observations_prompt(
        task_info: TaskInfo, evo_paras: CoEvoParas, parents: list, mode: str, num_observations: int
) -> tuple[dict[str, str], dict[str, str]]:
    sys_prompt = system_prompt_observation(task_info.program_lang, mode, len(parents))

    user_prompt = f"Here is the problem:\n\n{task_info.task_info}\n\n"

    for parent_i, each_parent in enumerate(parents):
        user_prompt += f"## No. {parent_i+1} Solution:\n\n"
        user_prompt += f"The observations used for generating the solution:\n\n"
        user_prompt += " - " + "\n\n - ".join(each_parent['obs']) + "\n\n"
        user_prompt += f"{evo_paras.rep_use.rep_name}:\n\n"
        user_prompt += f"{each_parent['sol'][evo_paras.rep_use.rep_name]}:\n\n"
        user_prompt += f"Evaluation result:\n\n"
        user_prompt += f"{each_parent['fitness_string']}\n\n"

    user_prompt += \
        f"Brainstorm a list of {num_observations} innovative and non-obvious observations about "
    prompt_content = {
        "crossover_positive": 'creating a new solution that has a totally different form from the given solutions but can be motivated from the existing ones. ',
        "crossover_negative": 'creating a new solution that has a totally different form from the given solutions. ',
        "mutation_positive": 'creating a solution in different forms but can be a modified version of the existing solution. ',
        "mutation_negative": 'creating a totally different solution from the existing solution. '
    }[mode]
    user_prompt += prompt_content
    user_prompt += \
        f"Before each observation, provide reasoning like this:\n\n" \
        f"Observation 1:\n" \
        f"[A paragraph or two of reasoning.]\n" \
        f"[Your observation] (e.g., 'XXX should be considered' or 'YYY can help with ZZZ.')\n\n" \
        f"Ensure observations are useful for solving the problem. Be concise and avoid restating the obvious."

    conversation = (
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    )
    return conversation

def get_impl_direct(mode, rep_use) -> str:
    prompt_content = {
        "crossover_positive": 'create a new solution that has a totally different form from the given solutions but can be motivated from the existing ones. ',
        "crossover_negative": 'create a new solution that has a totally different form from the given solutions. ',
        "mutation_positive": 'create a solution in different forms but can be a modified version of the existing solution. ',
        "mutation_negative": 'create a totally different solution from the existing solution. '
    }[mode]
    prompt = \
        f"Now, {prompt_content}\n" \
        f"Please ONLY return {rep_use.rep_def} inside markdown codeblocks."
    return prompt

def get_impl_indirect() -> str:
    prompt = \
        "Now, use the observations above to brainstorm a natural language solution to the problem above.\n" \
        "1. Go beyond intuition and generate simple, innovative ideas.\n" \
        "2. Before each step, quote EXACTLY the relevant part of the observations. This is crucial.\n"
    return prompt