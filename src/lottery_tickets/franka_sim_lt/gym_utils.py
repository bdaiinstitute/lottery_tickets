import franka_sim.envs  # noqa: F401 required import for franka sim envs
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from lottery_tickets.franka_sim_lt.wrappers.chunking import ChunkingWrapper
from lottery_tickets.franka_sim_lt.wrappers.obs import ObsWrapper


def make_frankasim_env(env_name, env_kwargs) -> gym.Env:
    env = gym.make(env_name, **env_kwargs)

    if not has_wrapper(env, ObsWrapper):
        # Flattens all non-image observations into one vector
        env = ObsWrapper(env)

    if not has_wrapper(env, ChunkingWrapper):
        env = ChunkingWrapper(
            env, obs_horizon=1, act_exec_horizon=None
        )  # Adds an observation-history dimension.

    if not has_wrapper(env, RecordEpisodeStatistics):
        # Prevent "ValueError: Attempted to add episode stats when they already exist."
        env = RecordEpisodeStatistics(env)

    return env


def has_wrapper(env, wrapper_class: type, max_depth: int = 10000) -> bool:
    """Check if the environment or any of its wrappers is an instance of the specified wrapper class.

    Args:
        env (gym.Env): The environment to check.
        wrapper_class (type): The wrapper class to look for.

    Returns:
        bool: True if the environment or any of its wrappers is an instance of the specified class, False otherwise.
    """
    current_env = env
    depth = 0
    while depth < max_depth:
        if isinstance(current_env, wrapper_class):
            return True
        if not hasattr(current_env, "env"):
            break
        current_env = current_env.env
        depth += 1
    return False
