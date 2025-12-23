# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections import deque
from typing import Optional, TypeAlias, Union

import gymnasium as gym
import numpy as np
import torch

NestedDict: TypeAlias = dict[str, Union[np.ndarray, "NestedDict"]]


def stack_obs(list_of_dicts: list[NestedDict]) -> NestedDict:
    """
    Stacks np.ndarrays in a list of nested dictionaries along a new batch dimension.
    From Gemini.

    Args:
        list_of_dicts: A list of nested dictionaries. The leaves of these
                       dictionaries must be np.ndarrays. All dictionaries
                       in the list must have the same schema.

    Returns:
        A new dictionary with the same schema as the input dictionaries, but
        with the leaf ndarrays stacked along a new batch dimension.
    """
    if not list_of_dicts:
        return {}

    def _recursive_stack(
        dicts: list[Union[NestedDict, np.ndarray]],
    ) -> Union[NestedDict, np.ndarray]:
        # Check if the first element is an ndarray, which signifies a leaf node
        if isinstance(dicts[0], np.ndarray):
            return np.stack(dicts, axis=0)  # type: ignore
        elif isinstance(dicts[0], torch.Tensor):
            return torch.stack(dicts, dim=0)

        # If it's a dictionary, recurse on its values
        result_dict: NestedDict = {}
        # We know dicts[0] is a Dict here
        for key in dicts[0].keys():
            # Create a list of the values for the current key from all dictionaries
            list_of_values_at_key = [d[key] for d in dicts]  # type: ignore
            result_dict[key] = _recursive_stack(list_of_values_at_key)  # type: ignore
        return result_dict

    return _recursive_stack(list_of_dicts)  # type: ignore


def space_stack(space: gym.Space, repeat: int):
    # Stack the observation or action space to match the chunked horizon
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError("Unsupported space type for stacking.")


class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        # Adjust observation and action spaces to account for chunking
        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon or 1
        action = [action] if act_exec_horizon == 1 else action

        assert len(action) >= act_exec_horizon

        # Execute actions for the defined horizon, appending observations
        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
            if done or trunc:
                break
        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs):
        # Reset and initialize the observation buffer with repeated initial obs
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info
