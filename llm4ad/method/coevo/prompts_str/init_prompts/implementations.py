def get_direct_impl_prompts(
        _task_description_str, coevo_paras, use_combo: tuple[str, ...]
):
    sys_prompt = f"You will be given a problem, its details and some important insights."
    user_prompt = get_impl_direct(_task_description_str, use_combo)
    conversation = (
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    )
    return conversation


def get_impl_direct(_task_description_str,  use_combo: tuple[str, ...]) -> str:
    prompt = f"Here is the problem:\n\n{_task_description_str}\n\n"
    if len(use_combo) == 0:
        prompt += "No observations are necessary to solve this problem.\n\n"
    else:
        prompt += "Here are the intelligent observations to help solve the problem:\n\n"
        prompt += " - " + "\n\n - ".join(use_combo) + "\n\n"
    prompt += f'''
1. First, describe your new implementation and main steps in one sentence. The description must be inside within boxed {{}}. 
2. Next, give the optimized kernel implementation:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.'''
    return prompt