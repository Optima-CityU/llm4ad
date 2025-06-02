import time
import random
import itertools
from typing import Any, Optional, Union, Callable




class ObservationNode:
    def __init__(
            self, task_info_inst, coevo_paras,
            layer_num: int, prev_observation_strs: tuple[str, ...] = ()
    ) -> None:
        self.task_info_inst = task_info_inst
        self.coevo_paras = coevo_paras

        self.layer_num = layer_num

        self.prev_observation_strs = prev_observation_strs
        self.derived_observations: Optional[list[str]] = None
        self.next_observation_nodes: Optional[list[ObservationNode]] = None

        self.num_observations_to_generate = self.coevo_paras.num_observations_to_generate
        self.max_observation_k = self.coevo_paras.max_observation_k

        self.log = None

        self.get_obs_time = None  # For later record

        assert self.layer_num >= 0
        if self.layer_num == 0:
            assert len(prev_observation_strs) == 0

    def get_observation(self, llm_inst):
        get_obs_time = time.time()

        # Getting Observations from LLM
        get_observations_prompt = CoEvoPrompt.get_init_observation_prompt(
            self.task_info_inst, self.coevo_paras, self.prev_observation_strs, self.layer_num
        )
        obs_str = llm_inst.get_response(get_observations_prompt)

        # From Observation String to List
        get_parse_into_list_prompt = get_observations_prompt + (
            {"role": "assistant", "content": obs_str},
            {"role": "user", "content": CoEvoPrompt.FORMAT_INTO_LIST_PROMPT}
        )
        obs_str_list = llm_inst.get_response(get_parse_into_list_prompt)

        # Filter out important Observations
        filter_observations_prompt = get_parse_into_list_prompt + (
            {"role": "assistant", "content": obs_str_list},
            {"role": "user", "content": CoEvoPrompt.FILTER_TO_USEFUL_LIST_PROMPT}
        )
        filtered_obs_str_list = llm_inst.get_response(filter_observations_prompt)

        # Parse into python lists
        parse_to_python_prompt = filter_observations_prompt + (
            {"role": "assistant", "content": filtered_obs_str_list},
            {"role": "user", "content": CoEvoPrompt.PARSE_INTO_PYTHON_LIST_PROMPT}
        )

        MAX_PARSE_TRIES = 3
        retry = 0
        while True:
            parsed_obs_python_list = llm_inst.get_response(parse_to_python_prompt)
            try:
                attempted_parse = eval(parsing_utils.markdown_codeblock_extract(parsed_obs_python_list))
                assert isinstance(attempted_parse, list)
                assert all(isinstance(parse, str) for parse in attempted_parse)
                self.get_obs_time = time.time() - get_obs_time
                self.log = parse_to_python_prompt + ({"role": "assistant", "content": parsed_obs_python_list},)
                self.derived_observations = tuple(attempted_parse)
                observation_combos = self.split_into_observation_combos(self.derived_observations)
                self.next_observation_nodes = []

                for combo in observation_combos:
                    new_node = self.create_node_like(combo)
                    self.next_observation_nodes.append(new_node)
                return
            except:
                retry += 1
                if retry >= MAX_PARSE_TRIES:
                    print("Parse Max Reached.")
                    exit(1)

    def get_highest_level_nodes(self, acc=None):
        if acc is None:
            acc = []

        if self.next_observation_nodes is None:
            acc.append(self)
            return acc

        for each_next in self.next_observation_nodes:
            acc = each_next.get_highest_level_nodes(acc)

        return acc

    def get_highest_level_combos(self):
        highest_level_nodes = self.get_highest_level_nodes()
        highest_level_combos = []
        for node in highest_level_nodes:
            highest_level_combos.append(node.prev_observation_strs)
        return highest_level_combos

    def split_into_observation_combos(self, observations: tuple[str, ...]) -> list[tuple[str, ...]]:
        max_k = min(self.max_observation_k, len(observations))

        observation_combos = []
        for k in range(max_k + 1):
            observation_combos.extend(itertools.combinations(observations, k))

        random.shuffle(observation_combos)
        return observation_combos


    def create_node_like(self, observation_combo: tuple[str, ...]) -> "ObservationNode":
        return ObservationNode(self.task_info_inst, self.coevo_paras, self.layer_num + 1, observation_combo)