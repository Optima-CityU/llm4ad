def get_summarizer_how_to():
    prompt_content = \
        f'As a smart analyzer, you will be presented with the task information and some solutions to the task, '\
        f'each with a list of ideas and their evaluation results.\n'\
        f'Your task is to to analyze why the last solution has the best performance by '\
        f'comparing the differences in the solutions\' ideas, implementations, and evaluation results.\n' \
        f'You will also receive an idea pool, which contains all the ideas you have already summarized.\n'
    return prompt_content

def get_summarizer_end():
    prompt_content = \
        f'Hints: You need to analyze why the last solution has the best performance and summarize the ideas that '\
        f'are effective for evaluation performance improvement in the format '\
        f'of: Reasoning (reasoning and analysis, why it is useful for solving the task or improving the performance), '\
        f'Name, Definition (brief description of the idea), and Example. Example part is extremely important for later reuse of the summarized idea, '\
        f'so make sure to provide a clear and concise example.\n\n'\
        f'Response Format: : (Replace ...)\n\n' \
        f'## New Ideas\n' \
        f'- Idea 1:\n' \
        f'  - Reasoning:...\n' \
        f'  - Name:...\n' \
        f'  - Definition:...\n' \
        f'  - Example:...\n\n' \
        f'...\n\n' \
        f'## Analysis\n\n' \
        f'...\n\n'
    return prompt_content