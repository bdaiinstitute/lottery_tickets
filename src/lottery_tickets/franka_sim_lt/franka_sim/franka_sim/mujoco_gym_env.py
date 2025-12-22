# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class GymRenderingSpec:
    """Rendering specifications for MujocoGymEnv."""
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        """
        Initializes a mujoco gym environment.

        Args:
            xml_path: Path to the Mujoco XML model.
            seed: The RNG seed to use.
            control_dt: The control timestep, in seconds.
            physics_dt: The physics timestep, in seconds.
            time_limit: Time limit on episode length, in seconds.
            render_spec: Rendering specifications, like height and width of image.
        """
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        """Method that renders the environment based on render_specs."""
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        """Closes the environment and the viewer if it exists."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def time_limit_exceeded(self) -> bool:
        """Checks if the time limit has been exceeded."""
        return self._data.time >= self._time_limit

    # Accessors.
    @property
    def model(self) -> mujoco.MjModel:
        """Accessor for the Mujoco model."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """Accessor for the Mujoco data."""
        return self._data

    @property
    def control_dt(self) -> float:
        """Accessor for the control timestep."""
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        """Accessor for the physics timestep."""
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        """Accessor for the RNG state."""
        return self._random
