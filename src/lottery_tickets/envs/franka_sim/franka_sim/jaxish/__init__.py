from franka_sim.jaxish.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register

register(
    id="JaxishPandaPickCube-v0",
    entry_point="franka_sim.jaxish.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="JaxishPandaPickCubeVision-v0",
    entry_point="franka_sim.jaxish.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="JaxishPandaPickCubeSparseVision-v0",
    entry_point="franka_sim.jaxish.envs:PandaPickCubeSparseGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
