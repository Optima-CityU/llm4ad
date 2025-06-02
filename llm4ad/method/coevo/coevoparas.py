import os.path
from typing import Literal


class CoEvoParas:
    def __init__(
            self,
            # Init ReAct Setting
            num_obs_layers: int = 2,
            num_observations_to_generate: int = 5,
            max_observation_k: int = 2,
            num_sol_from_a_tree: int = 5,

            # Single Solution Generation
            num_idea: list[int] = None,
            idea_unroll_mode: Literal["recent", "full"] = "recent",
    ):

        # ReAct Setting
        self.num_obs_layers = num_obs_layers
        self.num_observations_to_generate = num_observations_to_generate
        self.max_observation_k = max_observation_k
        self.num_sol_from_a_tree = num_sol_from_a_tree

        # Single Solution Generation
        self.num_idea = num_idea
        self.idea_unroll_mode = idea_unroll_mode