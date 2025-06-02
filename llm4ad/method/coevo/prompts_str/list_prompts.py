from coevo.methods.coevo.coevoparas import CoEvoParas

def list_pool(idea_pool):
    prompt_content = ""
    if len(idea_pool) == 0:
        return "{}\n"
    else:
        for each_inspiration_idx, each_inspiration in enumerate(idea_pool):
            prompt_content += \
                f'{each_inspiration_idx + 1}. {each_inspiration["Name"]}:\n'\
                f'  Definition: {each_inspiration["Definition"]}\n' \
                f'  Example: {each_inspiration["Example"]}\n\n'
        return prompt_content

def list_single_sequential(evo_paras: CoEvoParas, previous_result_dict_list: list[dict] = None):
    prompt_content = f""
    for prev_res_id, prev_res in enumerate(previous_result_dict_list):
        prompt_content += f'**Solution No.{prev_res_id + 1}**\n'
        prompt_content += "1. Its Ideas\n"
        for idea_idx, each_idea in enumerate(prev_res['Ideas']):
            prompt_content += f'- {each_idea["Name"]}: {each_idea["Definition"]}\n'
        prompt_content += \
            f'\n2. Its Format {evo_paras.rep_use.rep_name}:\n'\
            f'{prev_res["Solutions"][evo_paras.rep_use.rep_name]}\n\n'
        prompt_content += f'3. Its Evaluation Results\n'
        if prev_res["error_msg"] is not None:
            prompt_content += f'{prev_res["error_msg"]}\n\n'
        else:
            prompt_content += f'{prev_res["fitness_string"]}\n\n'
    return prompt_content

def list_parents(evo_paras: CoEvoParas, parents_list: list[dict]):
    indivs_prompt = f""
    for indiv_idx, indiv in enumerate(parents_list):
        indiv_res = parents_list[indiv_idx]
        indivs_prompt += \
            f'**Solution No.{indiv_idx + 1}**\n1. Its Ideas\n'
        for inspiration_idx, each_inspiration in enumerate(indiv_res['Ideas']):
            indivs_prompt += f'- {each_inspiration["Name"]}: {each_inspiration["Definition"]}\n'
        indivs_prompt += \
            f'\n2. Its Format {evo_paras.rep_use.rep_name}:\n' \
            f'{indiv_res["Solutions"][evo_paras.rep_use.rep_name]}\n\n'
        indivs_prompt += f'3. Its Evaluation Results\n'
        if indiv_res["error_msg"] is not None:
            indivs_prompt += f'{indiv_res["error_msg"]}\n\n'
        else:
            indivs_prompt += f'{indiv_res["fitness_string"]}\n\n'
    return indivs_prompt

def list_offsprings(evo_paras: CoEvoParas, offspring_list: list[dict], parents_num: int):
    indivs_prompt = f""

    for indiv_idx, indiv in enumerate(offspring_list):
        indiv_res = offspring_list[indiv_idx]
        indivs_prompt += \
            f'**Solution No.{indiv_idx + 1 + parents_num}** (Offspring from Solution No.1 - No.{parents_num})\n1. Its Ideas\n'
        for inspiration_idx, each_inspiration in enumerate(indiv_res['Ideas']):
            indivs_prompt += f'- {each_inspiration["Name"]}: {each_inspiration["Definition"]}\n'
        indivs_prompt += \
            f'\n2. Its Format {evo_paras.rep_use.rep_name}:\n' \
            f'{indiv_res["Solutions"][evo_paras.rep_use.rep_name]}\n\n'
        indivs_prompt += f'3. Its Evaluation Results\n'
        if indiv_res["error_msg"] is not None:
            indivs_prompt += f'{indiv_res["error_msg"]}\n\n'
        else:
            indivs_prompt += f'{indiv_res["fitness_string"]}\n\n'
    return indivs_prompt