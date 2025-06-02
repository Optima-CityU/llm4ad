from coevo.methods.coevo.coevoparas import CoEvoParas

def init_sol_response_format(evo_paras: CoEvoParas):
    prompt_content = \
        f'Response Format (Replace ...): \n\n' \
        f'## Ideas\n' \
        f'- Idea 1:\n' \
        f'  - Name:...\n' \
        f'  - Reasoning:...\n' \
        f'  - Definition:...\n...\n\n' \
        f'## Thoughts\n' \
        f'...\n\n' \
        f'## Solutions\n'
    for each_rep in evo_paras.rep_list:
        prompt_content += \
            f'### {each_rep.rep_name}:\n...\n'
    return prompt_content

def continue_sol_response_format(evo_paras: CoEvoParas):
    prompt_content = \
        f'Response Format: (Replace ...)\n\n' \
        f'## Ideas\n' \
        f'  - Idea 1:\n' \
        f'    - Quotes:...\n' \
        f'    - Implications:...\n'\
        f'    - Name:...\n' \
        f'    - Reasoning:...\n' \
        f'    - Definition:...\n' \
        f'...\n\n' \
        f'## Thoughts\n' \
        f'...\n\n' \
        f'## Solutions\n'
    for each_rep in evo_paras.rep_list:
        prompt_content += \
            f'### {each_rep.rep_name}:\n...\n'
    return prompt_content