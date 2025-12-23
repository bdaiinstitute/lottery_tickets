# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Shared utilities for noise evaluation scripts."""
import os
import json
import numpy as np


def evaluate_noise_single(env, noise_vec, save_vid=False, noise_idx=0, eval_num=0, rew_offset=1.0, initial_obs=None):
	"""Evaluate noise vector for one episode. Success if any step reward > -rew_offset.
	
	Args:
		env: Gym environment with diffusion policy wrapper
		noise_vec: Noise vector to evaluate
		save_vid: Whether to save video
		noise_idx: Index of the noise vector
		eval_num: Evaluation number for naming purposes
		rew_offset: Reward offset for success determination
		initial_obs: If provided, use this observation instead of calling env.reset()

	Returns:
		episode_reward: Total reward for the episode
		success: Boolean indicating if episode was successful
	"""
	if save_vid:
		env.env.name_prefix = f"noise_{noise_idx}_eval_{eval_num}"
	
	if initial_obs is None:
		env.reset()
	# else: use the provided initial_obs without resetting

	# Storage for single episode
	episode_reward = 0.0
	success = False
	done = False

	# Action is the noise vector (wrapper handles diffusion policy)
	action = noise_vec.reshape(1, -1)  # Shape (1, action_dim) for single env

	steps = 0
	while not done:
		_, r, d, info = env.step(action)
		steps += 1
		r_val = float(r[0])  # Extract scalar from array
		episode_reward += r_val
		
		if r_val > float(-rew_offset):  # terminated successfully
			success = True
		
		if d[0]:  # truncated or terminated
			done = True
			
	return episode_reward, success


def evaluate_gaussian_single(env, mean, cov, rng, save_vid=False, noise_idx=0, eval_num=0, rew_offset=1.0, initial_obs=None):
	"""Evaluate by sampling from Gaussian at every step. Success if any step reward > -rew_offset.
	
	Args:
		env: Gym environment with diffusion policy wrapper
		mean: Gaussian mean vector
		cov: Gaussian covariance matrix
		rng: numpy random generator
		save_vid: Whether to save video
		noise_idx: Index of the noise vector
		eval_num: Evaluation number for naming purposes
		rew_offset: Reward offset for success determination
		initial_obs: If provided, use this observation instead of calling env.reset()

	Returns:
		episode_reward: Total reward for the episode
		success: Boolean indicating if episode was successful
	"""
	if save_vid:
		env.env.name_prefix = f"noise_{noise_idx}_eval_{eval_num}"
	
	if initial_obs is None:
		env.reset()

	# Storage for single episode
	episode_reward = 0.0
	success = False
	done = False

	steps = 0
	while not done:
		# Sample from Gaussian at every step (no clipping)
		noise_vec = rng.multivariate_normal(mean, cov)
		action = noise_vec.reshape(1, -1)  # Shape (1, action_dim) for single env
		
		_, r, d, info = env.step(action)
		steps += 1
		r_val = float(r[0])  # Extract scalar from array
		episode_reward += r_val
		
		if r_val > float(-rew_offset):  # terminated successfully
			success = True
		
		if d[0]:  # truncated or terminated
			done = True
			
	return episode_reward, success


def save_eval_serial(out_dir, per_seed_rewards_all, per_seed_success_flags, episode_seeds, **kwargs):
	"""Save serial rollout results (n_seeds x n_evals_per_seed). Returns reward_mean, reward_std, success_mean, success_std.
	
	Args:
		out_dir: Output directory to save results
		per_seed_rewards_all: List of lists of rewards per seed
		per_seed_success_flags: List of lists of success flags per seed
		episode_seeds: List of episode seeds used
		**kwargs: Additional metadata to save in summary.json

	Returns:
		reward_mean: Mean reward across seeds
		reward_std: Std of reward across seeds
		success_mean: Mean success rate across seeds
		success_std: Std of success rate across seeds
	
	"""
	os.makedirs(out_dir, exist_ok=True)
	# Convert to arrays
	reward_matrix = np.array(per_seed_rewards_all, dtype=np.float32)  # shape (n_seeds, n_evals_per_seed)
	success_matrix = np.array(per_seed_success_flags, dtype=bool)     # shape (n_seeds, n_evals_per_seed)

	# Per-seed means (average across n_evals_per_seed for each seed)
	seed_reward_means = reward_matrix.mean(axis=1) if reward_matrix.size else np.array([])
	seed_success_means = success_matrix.mean(axis=1) if success_matrix.size else np.array([])

	# Overall stats (mean and std across all seeds)
	reward_mean = float(seed_reward_means.mean()) if seed_reward_means.size else 0.0
	reward_std = float(seed_reward_means.std(ddof=1)) if seed_reward_means.size else 0.0
	success_mean = float(seed_success_means.mean()) if seed_success_means.size else 0.0
	success_std = float(seed_success_means.std(ddof=1)) if seed_success_means.size else 0.0

	# Persist full data
	np.save(os.path.join(out_dir, "reward_matrix.npy"), reward_matrix)
	np.save(os.path.join(out_dir, "success_matrix.npy"), success_matrix)
	np.save(os.path.join(out_dir, "seed_reward_means.npy"), seed_reward_means)
	np.save(os.path.join(out_dir, "seed_success_means.npy"), seed_success_means)

	summary_data = {
		"n_seeds": int(reward_matrix.shape[0]) if reward_matrix.size else 0,
		"n_evals_per_seed": int(reward_matrix.shape[1]) if reward_matrix.size else 0,
		"reward_mean": reward_mean,
		"reward_std": reward_std,
		"success_mean": success_mean,
		"success_std": success_std,
		"episode_seeds": episode_seeds,
		"seed_reward_means": seed_reward_means.tolist(),
		"seed_success_means": seed_success_means.tolist(),
		"reward_matrix": reward_matrix.tolist(),
		"success_matrix": success_matrix.astype(bool).tolist(),
		**kwargs  # Add any additional metadata
	}

	with open(os.path.join(out_dir, "summary.json"), "w") as f:
		json.dump(summary_data, f)

	return reward_mean, reward_std, success_mean, success_std


def load_noise_idx(eval_path: str, eval_idx: int = 0):
	"""Load noise vector at eval_idx from noise_samples.npy in eval_path.
	
	Args:
		eval_path: Path to evaluation directory
		eval_idx: Index of noise vector to load
	
	Returns:
		noise_vec: Loaded noise vector
	"""
	noise_path = os.path.join(eval_path, "noise_samples.npy")
	if not os.path.exists(noise_path):
		raise FileNotFoundError(f"No noise_samples.npy found in {eval_path}")
	noise_samples = np.load(noise_path)
	if eval_idx >= len(noise_samples):
		raise IndexError(f"eval_idx={eval_idx} out of range, only {len(noise_samples)} noise samples available")
	return noise_samples[eval_idx]
