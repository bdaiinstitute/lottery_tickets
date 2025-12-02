import numpy as np

from franka_sim.jaxish.envs import PandaPickCubeGymEnv

REWARD_THRESHOLD = 0.8


class PandaPickCubeSparseGymEnv(PandaPickCubeGymEnv):
    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        rew = 0.3 * r_close + 0.7 * r_lift
        # Get a sparse reward of 1 if dense reward is above a threshold.
        return float(rew >= REWARD_THRESHOLD)
