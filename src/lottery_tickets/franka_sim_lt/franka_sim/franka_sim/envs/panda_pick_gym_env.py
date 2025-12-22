# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.utils import (
    close_gym_mjc_viewer_multiversion,
    create_gym_mjc_viewer_multiversion,
    render_gym_mjc_viewer_multiversion,
)

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_DEFAULT_SAMPLING_BOUNDS = np.asarray([[0.4, -0.0], [0.4, 0.0]])  # fixed reset


class PandaPickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        sampling_bounds=_DEFAULT_SAMPLING_BOUNDS,
    ):
        """
        Initializes a Panda pick cube gym environment.

        Args:
            action_scale: Scaling factors for the actions.
            seed: The RNG seed to use.
            control_dt: The control timestep, in seconds.
            physics_dt: The physics timestep, in seconds.
            time_limit: ???
            render_spec: ???
            render_mode: ???
            image_obs: If True, uses image observations. Else uses block state as observations.
            sampling_bounds: The sampling bounds for the block positions.
        """
        self._action_scale = action_scale
        self.sampling_bounds = np.asarray(sampling_bounds)
        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "panda/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/gripper_pos": spaces.Box(
                            -1, 1, shape=(1,), dtype=np.float32
                        ),
                        # "panda/joint_pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        # "panda/joint_vel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        # "panda/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
                        # "panda/wrist_force": specs.Array(shape=(3,), dtype=np.float32),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "panda/tcp_pos": spaces.Box(
                                np.float32(-np.inf),
                                np.float32(np.inf),
                                shape=(3,),
                                dtype=np.float32,
                            ),
                            "panda/tcp_vel": spaces.Box(
                                np.float32(-np.inf),
                                np.float32(np.inf),
                                shape=(3,),
                                dtype=np.float32,
                            ),
                            "panda/gripper_pos": spaces.Box(
                                -1.0, 1.0, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = create_gym_mjc_viewer_multiversion(
            model=self.model,
            data=self.data,
            height=render_spec.height,
            width=render_spec.width,
            camera_ids=self.camera_id,
        )
        self.render_images()

    def reset(
        self, seed=None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: The RNG seed to use for resetting.

        Returns:
            observation: dict[str, np.ndarray],
            info: dict[str, Any]
        """
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*self.sampling_bounds)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        x, y, z, grasp = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._compute_success()
        terminated = False  # use wrappers to handle time limit

        return obs, rew, terminated, False, {"is_success": success}

    def render(self) -> np.ndarray:
        """Render images. Concatenates images from multiple cameras. Always returns an array."""
        if self._viewer is None:
            raise ValueError("Viewer has not been initialized.")

        return np.concatenate(self.render_images(), axis=1)

    def render_images(self) -> list[np.ndarray]:
        """Standalone method for rendering images, which can be called directly by a user."""
        if self._viewer is None:
            raise ValueError("Viewer has not been initialized.")

        return render_gym_mjc_viewer_multiversion(
            viewer=self._viewer,
            render_mode=self.render_mode,
            camera_ids=self.camera_id,
        )

    def close(self) -> None:
        if self._viewer is not None:
            close_gym_mjc_viewer_multiversion(self._viewer)

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        # joint_pos = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_pos"] = joint_pos.astype(np.float32)

        # joint_vel = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_vel"] = joint_vel.astype(np.float32)

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        # joint_torque = np.stack(
        # [self._data.sensor(f"panda/joint{i}_torque").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_torque"] = symlog(joint_torque.astype(np.float32))

        # wrist_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
        # obs["panda/wrist_force"] = symlog(wrist_force.astype(np.float32))

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render_images()
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self.render_images()

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        rew = 0.3 * r_close + 0.7 * r_lift
        return rew

    def _compute_success(self) -> bool:
        # Note: untested
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        return dist < 0.01 and block_pos[2] > self._z_success


if __name__ == "__main__":
    # Note that render_mode="human" only works with gymnasium>=1.0.0
    # because we updated Mujoco.
    env = PandaPickCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render_images()
    env.close()
