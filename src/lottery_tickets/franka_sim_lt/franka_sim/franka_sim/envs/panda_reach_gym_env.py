# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np

from franka_sim.envs import PandaPickCubeGymEnv

REWARD_THRESHOLD = 0.8


class PandaReachCubeGymEnv(PandaPickCubeGymEnv):
    """Overrides PandaPickCubeGymEnv with a reward for reaching the cube with the end effector."""

    def _compute_reward(self) -> float:
        """Computes a reward for reaching a block, based on the distance between end effector and cube."""
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        return r_close
