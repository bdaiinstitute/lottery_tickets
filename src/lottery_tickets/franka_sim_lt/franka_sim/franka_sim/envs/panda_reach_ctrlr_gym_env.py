from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from franka_sim.controllers.opspace import opspace
from franka_sim.envs.panda_pick_gym_env import _CARTESIAN_BOUNDS, _PANDA_HOME
from franka_sim.envs.panda_reach_gym_env import PandaReachCubeGymEnv


class PandaReachCubeCtrlrGymEnv(PandaReachCubeGymEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Remove gripper action from action space
        # add damping_ratio to end of action space
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, float("inf")], dtype=np.float32),
            dtype=np.float32,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # x, y, z, grasp, damping_ratio = action
        x, y, z, damping_ratio = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # # Set gripper grasp.
        # g = self._data.ctrl[self._gripper_ctrl_id] / 255
        # dg = grasp * self._action_scale[1]
        # ng = np.clip(g + dg, 0.0, 1.0)
        # self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                damping_ratio=damping_ratio,
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = False  # use wrappers to handle time limit

        return obs, rew, terminated, False, {}
