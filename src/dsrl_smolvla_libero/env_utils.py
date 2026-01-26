"""Code assumes obs_steps is 1, as is the case for SmolVLA on Libero."""

import gym
import gymnasium
import numpy as np
import torch
from typing import Any
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors


def make_libero_env(task="libero_spatial", task_id=0, render=False):
    """Create a Libero environment using lerobot's LiberoEnv wrapper."""

    from libero.libero import get_libero_path
    from pathlib import Path

    # Get the task suite and task_suite_name
    task_suite_name = task  # e.g., 'libero_spatial'
    # Load the task suite
    benchmark_dict = get_libero_path(task_suite_name)
    task_suite = benchmark_dict["task_suite"]

    # Create LiberoEnv with the specified task and task_id
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",  # Use pixel + agent position observations (images + robot state)
        render_mode="rgb_array" if render else None,
    )
    return env


def _add_envs_task(
    env: gym.vector.VectorEnv, observation: dict[str, Any]
) -> dict[str, Any]:
    """Add task field to observations (required by SmolVLA)."""
    try:
        task_result = env.call("task_description")
        if isinstance(task_result, tuple):
            task_result = list(task_result)
        if not isinstance(task_result, list):
            raise TypeError(
                f"Expected task_description to return a list, got {type(task_result)}"
            )
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("All items in task_description result must be strings")
        observation["task"] = task_result
    except (AttributeError, TypeError):
        try:
            task_result = env.call("task")
            if isinstance(task_result, tuple):
                task_result = list(task_result)
            if not isinstance(task_result, list):
                raise TypeError(
                    f"Expected task to return a list, got {type(task_result)}"
                )
            if not all(isinstance(item, str) for item in task_result):
                raise TypeError("All items in task result must be strings")
            observation["task"] = task_result
        except (AttributeError, TypeError):
            num_envs = observation[list(observation.keys())[0]].shape[0]
            observation["task"] = ["" for _ in range(num_envs)]
    return observation


class ActionChunkWrapper(gymnasium.Env):
    """Wrapper for action chunking with reward aggregation and offset."""

    def __init__(self, env, cfg, max_episode_steps=300, fixed_seed=None):
        self.max_episode_steps = max_episode_steps
        self.env = env
        self.act_steps = cfg.policy.n_action_steps
        self.action_space = spaces.Box(
            low=np.tile(env.action_space.low, cfg.policy.n_action_steps),
            high=np.tile(env.action_space.high, cfg.policy.n_action_steps),
            dtype=np.float32,
        )
        self.observation_space = env.observation_space
        self.count = 0
        self.reward_offset = cfg.reward_offset
        self._episode_success = (
            False  # track success across chunked steps until termination
        )

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.count = 0
        self._episode_success = False
        return obs, {}

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(self.act_steps, -1)
        obs_ = []
        reward_ = []
        done_ = []
        info_ = []
        for i in range(self.act_steps):
            self.count += 1
            obs_i, reward_i, done_i, info_i = self.env.step(action[i])
            obs_.append(obs_i)
            reward_.append(reward_i)
            done_.append(done_i)
            info_.append(info_i)
        obs = obs_[-1]
        reward = float(sum(reward_)) - self.reward_offset  # aggregate chunk rewards
        info = info_[-1]
        # Success criterion based on aggregated summed reward exceeding threshold
        if reward > float(-self.reward_offset):
            self._episode_success = True
        # underlying env signalled termination in any inner step
        underlying_done = bool(np.max(done_))
        terminated = self._episode_success or underlying_done
        truncated = False
        if self.count >= self.max_episode_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
        # annotate success for logging callbacks
        info["is_success"] = bool(self._episode_success)
        if terminated or truncated:
            info["terminal_observation"] = obs
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return


class SmolVLAPolicyEnvWrapper(VecEnvWrapper):
    """Wrapper that uses SmolVLA policy for action prediction in DSRL training.

    Handles dict observations from LiberoEnv (images + robot state) and:
    - Applies preprocessor to observations (preprocess_observation + env-specific + policy-specific)
    - Uses SmolVLA policy for action prediction with noise injection
    - Applies postprocessor to actions before passing to environment
    - Exposes processed observations to SAC's MultiInputPolicy with LiberoFeatureExtractor
    """

    def __init__(self, env, cfg, base_policy):
        super().__init__(env)
        self.base_policy = base_policy  # SmolVLA policy
        self.action_horizon = cfg.policy.n_action_steps
        self.action_dim = base_policy.config.max_action_dim
        self.device = base_policy.device

        # Create preprocessors and postprocessors following evaluate.py pattern
        preprocessor_overrides = {
            "device_processor": {"device": str(base_policy.config.device)},
            "rename_observations_processor": {
                "rename_map": getattr(cfg, "rename_map", {})
            },
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO)
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

        # Action space for the noise (what DSRL agent outputs)
        self.action_space = spaces.Box(
            low=-cfg.train.action_magnitude
            * np.ones(self.action_dim * self.action_horizon),
            high=cfg.train.action_magnitude
            * np.ones(self.action_dim * self.action_horizon),
            dtype=np.float32,
        )

        # Observation space - keep the dict space from LiberoEnv
        self.observation_space = env.observation_space
        self.env = env
        self.obs = None

    def step_async(self, actions):
        # Reshape noise to (n_envs, action_horizon, action_dim)
        noise_tensor = torch.tensor(
            actions, device=self.device, dtype=torch.float32
        ).view(-1, self.action_horizon, self.action_dim)
        # Apply env_preprocessor and policy preprocessor
        processed_obs = self.env_preprocessor(self.obs)
        processed_obs = self.preprocessor(processed_obs)

        # Use SmolVLA's predict_action_chunk with the noise
        with torch.no_grad():
            action_chunk = self.base_policy.predict_action_chunk(
                processed_obs, noise=noise_tensor
            )

        # Apply postprocessing to actions
        action_transition = {"action": action_chunk}
        action_transition = self.postprocessor(action_transition)
        action_transition = self.env_postprocessor(action_transition)
        action_chunk = action_transition["action"]

        # Flatten to (n_envs, chunk_size * action_dim) for ActionChunkWrapper
        actions_flat = action_chunk.reshape(action_chunk.shape[0], -1).cpu().numpy()
        self.venv.step_async(actions_flat)

    def step_wait(self):
        """Wait for the environment step, preprocess observations and update stored obs."""
        obs, rewards, dones, infos = self.venv.step_wait()
        # Apply generic preprocessing and add task fields
        obs = preprocess_observation(obs)
        obs = _add_envs_task(self.venv, obs)
        self.obs = obs
        return obs, rewards, dones, infos

    def reset(self):
        """Reset the environment, preprocess observations and store."""
        obs = self.venv.reset()
        # Apply generic preprocessing and add task fields
        obs = preprocess_observation(obs)
        obs = _add_envs_task(self.venv, obs)
        self.obs = obs
        return obs
