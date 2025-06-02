def get_offspring_how_to(mode: str):
    prompt_content = "You will be presented with some existing solutions to the task. "
    prompt_content += {
        "crossover_positive": 'Now create a new solution that has a totally different form from the given solutions but can be motivated from the existing ones. ',
        "crossover_negative": 'Now create a new solution that has a totally different form from the given solutions. ',
        "mutation_positive": 'Now create a solution in different forms but can be a modified version of the existing solution. ',
        "mutation_negative": 'Now create a totally different solution from the existing solution. '
    }[mode]

    return prompt_content

def get_offspring_idea_hints(mode: str):
    return {
        "crossover_positive": ' The ideas should be diverse but can be motivated from the existing solutions\' ideas. ',
        "crossover_negative": ' The ideas should be totally different from the existing solutions\' ideas. ',
        "mutation_positive": ' The ideas should be in different form but can be a modified version of the existing solution\'s ideas. ',
        "mutation_negative": ' The ideas should be totally different from the existing solution\'s ideas.'
    }[mode]