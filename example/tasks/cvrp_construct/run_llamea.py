import os

# LLM backend – Gemini is used here, but the library supports OpenAI, Ollama and DeepSeek.
from llm4ad.tools.llm.llm_api_https import HttpsApi

# LLaMEA for LLM4AD evolutionary method.
from llm4ad.method.llamea import LLaMEA
from llm4ad.method.llamea.llamea_llm import LlameaLLM

# Example optimization task – Capacitated Vehicle Routing Problem (CVRP)
from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation
from llm4ad.task.optimization.cvrp_construct.template import (
    task_description,
    template_program,
)


def main():
    """
    Example: Run LLaMEA on the CVRP optimization task using Gemini as LLM.

    LLaMEA combines genetic algorithms with large language models to evolve programs or
    solutions. The LLM generates new solution variations while the evaluator scores them.
    """

    # --- 1. Setup LLM backend ---
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise RuntimeError(
    #         "Missing GOOGLE_API_KEY environment variable. Please export it first."
    #     )
    llm = LlameaLLM(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                    key='sk-xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                    model='xxx')  # your llm, e.g., 'gpt-3.5-turbo'; timeout defaults to 60


    # --- 2. Define problem evaluator ---
    # CVRPEvaluation evaluates candidate solutions for the CVRP task.
    evaluator = CVRPEvaluation()

    # --- 3. Define prompts used by LLaMEA ---
    # role_prompt tells the LLM who it is supposed to be
    role_prompt = (
        "You are a highly skilled researcher in metaheuristics and optimization. "
        "You design efficient algorithms to solve combinatorial optimization problems."
    )

    # task_description = plain English problem explanation
    # template_program = starter solution code prompt

    # --- 4. Initialize LLaMEA optimizer for (1+1) optimisation ---
    method = LLaMEA(
        llm=llm,
        evaluator=evaluator,
        n_parents=1,
        n_offsprings=1,
        iterations=10,           # Keep small for demonstration
        minimization=False,      # CVRP Evaluator here is maximization by default
        elitism=True,            # Keep best solution each generation
        role_prompt=role_prompt,
        task_prompt=task_description,
        example_prompt=template_program,
    )

    # --- 5. Run the optimization ---
    result = method.run()

    # --- 6. View result ---
    print("\nBest Solution Found:")
    print(f"Code\n{result.code}")
    print(f"with average path distances: {-result.fitness}.")


if __name__ == "__main__":
    main()
