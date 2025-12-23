# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten, flatten_space


class ObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.image_obs = "images" in list(self.env.observation_space.keys())
        self.has_state = "state" in set(self.env.observation_space.keys())
        obs_dict = dict()
        if self.has_state:
            obs_dict["state"] = flatten_space(self.env.observation_space["state"])
            obs_dict["state"].dtype = np.float32
        obs_dict.update(
            **(self.env.observation_space["images"] if self.image_obs else {}),
        )
        self.observation_space = gym.spaces.Dict(obs_dict)
        self.orig_observation_space = self.env.observation_space

    def observation(self, observation):
        obs = dict()
        if self.has_state:
            obs["state"] = flatten(
                self.orig_observation_space["state"], observation["state"]
            )
            obs["state"] = obs["state"].astype(np.float32)
        obs.update(
            **(observation["images"] if self.image_obs else {}),
        )
        return obs
