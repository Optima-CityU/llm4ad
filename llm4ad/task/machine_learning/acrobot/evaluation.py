# name: str: CarMountain
# Parameters:
# max_steps: int: 500
# end
from __future__ import annotations

from typing import Any
import gym

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.acrobot.template import template_program, task_description

__all__ = ['ACROBOT']

def evaluate(env: gym.Env, action_select: callable) -> float:
    """Evaluate heuristic function on car mountain problem."""

    observation, _ = env.reset()  # initialization
    action = 0  # initial action

    for i in range(env._max_episode_steps + 1):  # protect upper limits
        action = action_select(observation[0],
                               observation[1],
                               observation[2],
                               observation[3],
                               observation[4],
                               observation[5],
                               action)
        observation, reward, done, truncated, info = env.step(action)

        if done or truncated:
            # self.env.close()
            fitness = observation[0] + (observation[0] * observation[2] - observation[1] * observation[3]) + 2
            if fitness <= 1:
                return -(i + 1) / env._max_episode_steps
            else:
                return -fitness


class ACROBOT(Evaluation):
    """Evaluator for car mountain problem."""

    def __init__(self, max_steps=500, **kwargs):
        """
            Args:
                - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
                - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

            Attributes:
                - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )

        self.env = None
        self.env = gym.make('Acrobot-v1')
        self.env._max_episode_steps = max_steps

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self.env, callable_func)
