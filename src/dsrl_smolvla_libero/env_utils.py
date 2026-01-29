"""Code assumes obs_steps is 1, as is the case for SmolVLA on Libero."""

import cv2
import gym
import gymnasium
import numpy as np
import torch
from typing import Any
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from lerobot.envs.libero import LiberoEnv, _parse_camera_names
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors


def _stack_nested_obs(obs_list):
    """Recursively stack nested dict observations from multiple envs."""
    if len(obs_list) == 0:
        return obs_list

    first = obs_list[0]

    if isinstance(first, dict):
        return {
            key: _stack_nested_obs([obs[key] for obs in obs_list])
            for key in first.keys()
        }
    elif isinstance(first, (tuple, list)):
        return type(first)(
            _stack_nested_obs([obs[i] for obs in obs_list]) for i in range(len(first))
        )
    elif isinstance(first, np.ndarray):
        return np.stack(obs_list)
    else:
        # Scalar or other type - try to make array
        return np.array(obs_list)


def make_libero_env(env_cfg, task_id=0, render=False):
    """Create a Libero environment using lerobot's LiberoEnv with config."""
    from libero.libero import benchmark

    task_suite_name = env_cfg.task
    bench = benchmark.get_benchmark_dict()
    if task_suite_name not in bench:
        raise ValueError(
            f"Unknown LIBERO suite '{task_suite_name}'. Available: {', '.join(sorted(bench.keys()))}"
        )

    task_suite = bench[task_suite_name]()

    if not getattr(task_suite, "tasks", None):
        raise ValueError(f"Suite '{task_suite_name}' has no tasks.")

    return LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array" if render else None,
        camera_name=_parse_camera_names(env_cfg.camera_name),
        init_states=env_cfg.init_states,
        episode_length=env_cfg.episode_length,
        control_mode=env_cfg.control_mode,
    )


class ActionChunkWrapper(gymnasium.Env):
    """Wrapper for action chunking with reward aggregation and offset.

    Also flattens nested dict observations to avoid SB3's nested space check.
    Original observations are stored in info['nested_obs'] for SmolVLA.
    """

    def __init__(
        self, env, cfg, max_episode_steps=300, fixed_seed=None, chunk_size=None
    ):
        self.max_episode_steps = max_episode_steps
        self.env = env
        # Use chunk_size if provided (for SmolVLA), otherwise fall back to n_action_steps
        self.act_steps = (
            chunk_size if chunk_size is not None else cfg.policy.n_action_steps
        )
        self.action_space = spaces.Box(
            low=np.tile(env.action_space.low, self.act_steps),
            high=np.tile(env.action_space.high, self.act_steps),
            dtype=np.float32,
        )

        # Create flattened observation space for SB3 compatibility
        self._orig_obs_space = env.observation_space
        self.observation_space = self._create_flat_obs_space(env.observation_space)

        self.count = 0
        self.reward_offset = cfg.reward_offset
        self._episode_success = (
            False  # track success across chunked steps until termination
        )
        # Store task description (language instruction) for SmolVLA
        self._task = getattr(env, "task_description", getattr(env, "task", ""))

    def _create_flat_obs_space(self, obs_space):
        """Create a flat observation space from nested dict space with 64x64 images."""
        flat_space = {}

        # Flatten pixels and resize to 64x64: pixels.image -> image
        if "pixels" in obs_space.spaces:
            for img_key, img_space in obs_space["pixels"].spaces.items():
                # Create new space with 64x64 dimensions (channels, 64, 64)
                n_channels = (
                    img_space.shape[-1]
                    if img_space.shape[-1] in [1, 3, 4]
                    else img_space.shape[0]
                )
                flat_space[img_key] = spaces.Box(
                    low=0, high=255, shape=(n_channels, 64, 64), dtype=np.uint8
                )

        # Flatten robot_state into a single state vector
        if "robot_state" in obs_space.spaces:
            state_dim = 0
            robot_state_space = obs_space["robot_state"]
            for group_key in ["eef", "gripper", "joints"]:
                if group_key in robot_state_space.spaces:
                    for _, subspace in robot_state_space[group_key].spaces.items():
                        state_dim += int(np.prod(subspace.shape))

            flat_space["state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
            )

        return spaces.Dict(flat_space)

    def _flatten_obs(self, obs):
        """Flatten nested observation dict and resize images to 64x64."""
        flat = {}

        # Extract and resize images to 64x64
        if "pixels" in obs:
            for img_key, img_val in obs["pixels"].items():
                img = np.asarray(img_val)
                # Handle channel ordering - convert to HWC for cv2 if needed
                if img.shape[0] in [1, 3, 4] and img.ndim == 3:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
                # Resize to 64x64
                img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                # Convert back to CHW format
                if img_resized.ndim == 2:
                    img_resized = img_resized[np.newaxis, :, :]
                else:
                    img_resized = np.transpose(img_resized, (2, 0, 1))
                flat[img_key] = img_resized.astype(np.uint8)

        # Extract and concatenate state
        if "robot_state" in obs:
            state_parts = []
            robot_state = obs["robot_state"]
            for group_key in ["eef", "gripper", "joints"]:
                if group_key in robot_state:
                    for _, val in sorted(robot_state[group_key].items()):
                        val_flat = np.asarray(val).flatten()
                        state_parts.append(val_flat)
            if state_parts:
                flat["state"] = np.concatenate(state_parts)

        return flat

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.count = 0
        self._episode_success = False
        # Store original nested obs for SmolVLA
        info["nested_obs"] = obs
        info["task"] = self._task
        return self._flatten_obs(obs), info

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(self.act_steps, -1)
        obs_ = []
        reward_ = []
        terminated_ = []
        truncated_ = []
        info_ = []
        for i in range(self.act_steps):
            self.count += 1
            obs_i, reward_i, terminated_i, truncated_i, info_i = self.env.step(
                action[i]
            )
            obs_.append(obs_i)
            reward_.append(reward_i)
            terminated_.append(terminated_i)
            truncated_.append(truncated_i)
            info_.append(info_i)
        obs = obs_[-1]
        reward = float(sum(reward_)) - self.reward_offset  # aggregate chunk rewards
        info = info_[-1]
        # Success criterion based on aggregated summed reward exceeding threshold
        if reward > float(-self.reward_offset):
            self._episode_success = True
        # underlying env signalled termination in any inner step
        underlying_terminated = bool(np.max(terminated_))
        underlying_truncated = bool(np.max(truncated_))
        terminated = self._episode_success or underlying_terminated
        truncated = underlying_truncated
        if self.count >= self.max_episode_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
        # annotate success for logging callbacks
        info["is_success"] = bool(self._episode_success)
        # Store original nested obs for SmolVLA
        info["nested_obs"] = obs
        info["task"] = self._task
        if terminated or truncated:
            info["terminal_observation"] = self._flatten_obs(obs)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return


class SmolVLAPolicyEnvWrapper(VecEnvWrapper):
    """Wrapper that uses SmolVLA policy for action prediction in DSRL training.

    Handles dict observations from LiberoEnv (images + robot state) and:
    - Retrieves nested observations from info['nested_obs'] (flattened by ActionChunkWrapper)
    - Applies preprocessor to observations (preprocess_observation + env-specific + policy-specific)
    - Uses SmolVLA policy for action prediction with noise injection
    - Applies postprocessor to actions before passing to environment
    - Passes through flattened observations to SAC

    Note: obs_steps=1 for SmolVLA, so we only need the most recent observation.
    """

    def __init__(self, env, cfg, base_policy):
        super().__init__(env)
        self.base_policy = base_policy  # SmolVLA policy
        # Use chunk_size for the full action chunk, not n_action_steps (which is for execution)
        self.action_horizon = base_policy.config.chunk_size  # 50 for SmolVLA
        self.action_dim = base_policy.config.max_action_dim  # 32 for LIBERO
        self.device = "cuda:0"
        self.n_envs = env.num_envs
        self.noise_shrink = (
            cfg.noise_shrink
        )  # If True, noise is sampled for action_dim only

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
        # If noise_shrink is True, action space is only action_dim (noise replicated across chunk)
        noise_space_dim = (
            self.action_dim
            if self.noise_shrink
            else self.action_dim * self.action_horizon
        )
        self.action_space = spaces.Box(
            low=-cfg.train.action_magnitude * np.ones(noise_space_dim),
            high=cfg.train.action_magnitude * np.ones(noise_space_dim),
            dtype=np.float32,
        )

        # Observation space is already flat from ActionChunkWrapper
        self.observation_space = env.observation_space
        self.obs = None  # Stores preprocessed obs for SmolVLA

    def _preprocess_obs(self, nested_obs_list, task_list):
        """Preprocess observations for SmolVLA (obs_steps=1, just stack along batch)."""
        # Stack observations along batch dimension
        stacked = _stack_nested_obs(nested_obs_list)
        # Apply generic preprocessing (converts to tensors, normalizes images, etc.)
        processed = preprocess_observation(stacked)
        # Add task
        processed["task"] = task_list
        return processed

    def step_async(self, actions):
        # Reshape noise to (n_envs, action_horizon, action_dim)
        if self.noise_shrink:
            # Actions are (n_envs, action_dim), replicate across action_horizon
            noise_single = torch.tensor(
                actions, device=self.device, dtype=torch.float32
            ).view(-1, self.action_dim)
            # Use .clone() after expand to create contiguous memory (SmolVLA does in-place ops)
            noise_tensor = (
                noise_single.unsqueeze(1).expand(-1, self.action_horizon, -1).clone()
            )
        else:
            noise_tensor = torch.tensor(
                actions, device=self.device, dtype=torch.float32
            ).view(-1, self.action_horizon, self.action_dim)

        # Apply env_preprocessor and policy preprocessor to stored observations
        processed_obs = self.env_preprocessor(self.obs)
        processed_obs = self.preprocessor(processed_obs)

        # Use SmolVLA's predict_action_chunk with the noise
        with torch.no_grad():
            action_chunk = self.base_policy.predict_action_chunk(
                processed_obs, noise=noise_tensor
            )

        # Apply postprocessing to actions
        # postprocessor expects just the action tensor, env_postprocessor expects a dict
        action_chunk = self.postprocessor(action_chunk)
        action_transition = {"action": action_chunk}
        action_transition = self.env_postprocessor(action_transition)
        action_chunk = action_transition["action"]

        # Flatten to (n_envs, chunk_size * action_dim) for ActionChunkWrapper
        actions_flat = action_chunk.reshape(action_chunk.shape[0], -1).cpu().numpy()
        self.venv.step_async(actions_flat)

    def step_wait(self):
        """Wait for the environment step and extract nested observations from infos."""
        flat_obs, rewards, dones, infos = self.venv.step_wait()

        # Extract nested observations and tasks from infos
        nested_obs_list = [info.get("nested_obs", {}) for info in infos]
        task_list = [info.get("task", "") for info in infos]

        # Preprocess for SmolVLA
        self.obs = self._preprocess_obs(nested_obs_list, task_list)

        return flat_obs, rewards, dones, infos

    def reset(self):
        """Reset the environment and extract nested observations from infos."""
        flat_obs = self.venv.reset()

        # Reset SmolVLA's internal state (queues, etc.)
        self.base_policy.reset()

        # Get nested obs from DummyVecEnv's reset_infos
        nested_obs_list = []
        task_list = []
        reset_infos = getattr(self.venv, "reset_infos", None)
        if reset_infos:
            nested_obs_list = [info.get("nested_obs", {}) for info in reset_infos]
            task_list = [info.get("task", "") for info in reset_infos]

        # Preprocess for SmolVLA
        self.obs = self._preprocess_obs(nested_obs_list, task_list)

        return flat_obs
