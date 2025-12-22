# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.envs.panda_pick_sparse_gym_env import PandaPickCubeSparseGymEnv
from franka_sim.envs.panda_reach_ctrlr_gym_env import PandaReachCubeCtrlrGymEnv
from franka_sim.envs.panda_reach_gym_env import PandaReachCubeGymEnv

__all__ = [
    "PandaReachCubeGymEnv",
    "PandaReachCubeCtrlrGymEnv",
    "PandaPickCubeGymEnv",
    "PandaPickCubeSparseGymEnv",
]

from gymnasium.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeSparse-v0",
    entry_point="franka_sim.envs:PandaPickCubeSparseGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaReachCube-v0",
    entry_point="franka_sim.envs:PandaReachCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaReachCubeVision-v0",
    entry_point="franka_sim.envs:PandaReachCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPickCubeSparseVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeSparseGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPickCubeRealisticControl-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={
        "image_obs": False,
        "control_dt": 0.1,
        "physics_dt": 0.002,
        "action_scale": np.asarray([0.05, 1.0]),
    },
)
register(
    id="PandaPickCubeVisionRealisticControl-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={
        "image_obs": True,
        "control_dt": 0.1,
        "physics_dt": 0.002,
        "action_scale": np.asarray([0.05, 1.0]),
    },
)
register(
    id="PandaReachCubeCtrlr-v0",
    entry_point="franka_sim.envs:PandaReachCubeCtrlrGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": False},
)
