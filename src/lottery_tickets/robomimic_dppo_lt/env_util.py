# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json

import gym
import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecVideoRecorder

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def make_robomimic_env(render=False, env="square", normalization_path=None, low_dim_keys=None, dppo_path=None, reward_shaping=False):
	wrappers = OmegaConf.create({
		"robomimic_lowdim": {
			"normalization_path": normalization_path,
			"low_dim_keys": low_dim_keys,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f"{dppo_path}/cfg/robomimic/env_meta/{env}.json"
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["env_kwargs"]["reward_shaping"] = reward_shaping
	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		# options = kwargs.get("options", {})
		# new_seed = options.get("seed", None)
		new_seed = kwargs.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs["state"].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs["state"].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()	


class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300, fixed_seed=None):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.ones(cfg.obs_dim),
			high=np.ones(cfg.obs_dim),
			dtype=np.float32
		)
		self.count = 0
		self.ns_seed = int(fixed_seed) if fixed_seed is not None else None
		self.reward_offset = getattr(cfg.env, "reward_offset", 1.0)
		self._episode_success = False  # track success across chunked steps until termination

	# Implemented for noise-search
	def reset(self, seed=None):
		if self.ns_seed is not None:
			obs = self.env.reset(seed=int(self.ns_seed))
		else:
			obs = self.env.reset(seed=seed)
		self.count = 0
		self._episode_success = False
		return obs, {}
	
	def step(self, action):
		# ------------------------------------------------------------
		# Original implementation
		# Kept here for reference; it used a single 'done' boolean and
		# did not distinguish termination vs truncation nor expose success.
		# ------------------------------------------------------------
		# if len(action.shape) == 1:
		# 	action = action.reshape(self.act_steps, -1)
		# obs_ = []
		# reward_ = []
		# done_ = []
		# info_ = []
		# done_i = False
		# for i in range(action.shape[0]):
		# 	self.count += 1
		# 	obs_i, reward_i, done_i, info_i = self.env.step(action[i])
		# 	obs_.append(obs_i)
		# 	reward_.append(reward_i)
		# 	done_.append(done_i)
		# 	info_.append(info_i)
		# obs = obs_[-1]
		# reward = sum(reward_)
		# done = np.max(done_)
		# info = info_[-1]
		# if self.count >= self.max_episode_steps:
		# 	done = True
		# if done:
		# 	info["terminal_observation"] = obs
		# return obs, reward, done, False, info
		# ------------------------------------------------------------
		# Updated implementation below adds success flag and separates
		# terminated vs truncated following SB3 guidelines
		# ------------------------------------------------------------
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = float(sum(reward_))  # aggregate chunk rewards
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
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	"""Wrapper that feeds either continuous noise trajectories or discrete noise library entries to a diffusion policy.

	If a noise_library (np.ndarray of shape (K, action_horizon*action_dim)) is provided, the action space becomes Discrete(K)
	and incoming integer actions index into the library. Otherwise a continuous Box space is exposed identical to prior implementation.
	"""
	def __init__(self, env, cfg, base_policy, noise_library: np.ndarray | None = None):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.discrete_mode = noise_library is not None
		self.noise_library = noise_library  # flattened noise vectors (K, horizon*action_dim) or None
		if self.discrete_mode:
			self.action_space = spaces.Discrete(len(self.noise_library))
		else:
			self.action_space = spaces.Box(
				low=-cfg.train.action_magnitude * np.ones(self.action_dim * self.action_horizon),
				high=cfg.train.action_magnitude * np.ones(self.action_dim * self.action_horizon),
				dtype=np.float32,
			)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim), 
			dtype=np.float32,
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None
		self.max_episode_steps = cfg.env.max_episode_steps

	def step_async(self, actions):
		if self.discrete_mode:
			# actions expected shape (n_env,) int indices
			if isinstance(actions, np.ndarray):
				indices = actions.reshape(-1)
			else:
				indices = np.array(actions).reshape(-1)
			noise_flat = self.noise_library[indices]  # (n_env, horizon*action_dim)
			actions_tensor = torch.tensor(noise_flat, device=self.device, dtype=torch.float32).view(-1, self.action_horizon, self.action_dim)
		else:
			actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32).view(-1, self.action_horizon, self.action_dim)
		diffused_actions = self.base_policy(self.obs, actions_tensor)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		# this done is truncated or terminated, set in _worker of SB3
		obs, rewards, dones, infos = self.venv.step_wait()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()


def build_lt_env(policy, cfg, n_envs, video_dir, save_vid=True, fixed_seeds=None, reward_shaping=False):
	"""
	Build vectorized environment with optional fixed seeds per env.
	IMP: This is for lottery ticket search where each env must have a fixed seed across resets.
	"""
	def make_env_fn(seed=None):
		def f():
			env = make_robomimic_env(
				env=cfg.env_name,
				normalization_path=cfg.normalization_path,
				low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
				dppo_path=cfg.dppo_path,
				render=save_vid,
				reward_shaping=reward_shaping
			)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
			env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps, fixed_seed=seed)
			return env
		return f

	if fixed_seeds is not None:
		env_fns = [make_env_fn(fixed_seeds[i]) for i in range(n_envs)]
	else:
		env_fns = [make_env_fn() for _ in range(n_envs)]
	
	if n_envs > 1:
		env = SubprocVecEnv(env_fns, start_method="spawn")
	else:
		env = DummyVecEnv(env_fns)
	
	setattr(env, "render_mode", "rgb_array")
	if save_vid:
		env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda step: step % 1 == 0, video_length=500)
	
	env = DiffusionPolicyEnvWrapper(env, cfg, policy)
	return env


def build_single_env(policy, cfg, video_dir, seed, save_vid=True):
	"""Build single environment for serial rollouts."""
	def make_env_fn():
		def f():
			env = make_robomimic_env(
				env=cfg.env_name,
				normalization_path=cfg.normalization_path,
				low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
				dppo_path=cfg.dppo_path,
				render=save_vid
			)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
			env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
			return env
		return f
	
	env = DummyVecEnv([make_env_fn()])	
	setattr(env, "render_mode", "rgb_array")
	if save_vid:
		env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda step: step % 1 == 0, video_length=400)
	
	env = DiffusionPolicyEnvWrapper(env, cfg, policy)
	env.seed(seed)
	return env


def build_single_env_with_reward_shaping(base_policy, cfg, video_dir, seed, save_vid=False):
	"""Build single environment with reward shaping enabled for multi-stage evaluation."""
	def make_env_fn():
		def f():
			env = make_robomimic_env(
				env=cfg.env_name,
				normalization_path=cfg.normalization_path,
				low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
				dppo_path=cfg.dppo_path,
				render=save_vid,
				reward_shaping=True  # Enable dense rewards for sub-task switching
			)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
			env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
			return env
		return f
	
	env = DummyVecEnv([make_env_fn()])
	setattr(env, "render_mode", "rgb_array")
	if save_vid:
		env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda step: step % 1 == 0, video_length=500)
	env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(seed)
	return env
