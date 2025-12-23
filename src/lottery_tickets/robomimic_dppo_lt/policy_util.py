# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
from typing import Any

from omegaconf import DictConfig
import hydra
import numpy as np
import torch
import wandb

from model.diffusion.diffusion import DiffusionModel
from stable_baselines3.common.callbacks import BaseCallback


class DPPOBasePolicyWrapper:
	"""
	Wraps a DPPO Base Policy and provides a call method.
	"""
	def __init__(self, base_policy: DiffusionModel):
		"""Initializes a wrapper policy.
		
		Args:
			base_policy: The base diffusion policy to wrap.
		"""
		self.base_policy = base_policy

	def __call__(self, obs: torch.Tensor, initial_noise: torch.Tensor, return_numpy: bool = True):
		"""
		Generates actions from the base policy given observations and initial noise.

		Args:
			obs: The observations tensor.
			initial_noise: The initial noise tensor.
			return_numpy: Whether to return actions as numpy array.

		Returns:
			The generated actions, either as a torch tensor or numpy array.
		"""
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg: DictConfig) -> DPPOBasePolicyWrapper:
	"""Loads a base policy from a Hydra configuration and returns a policy wrapper instance.
	
	Args:
		cfg: The Hydra configuration for the policy.

	Returns:
		A DPPOBasePolicyWrapper instance containing the loaded base policy.
	"""
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	print(f">>> Is base policy noise controllable: {base_policy.controllable_noise}")
	return DPPOBasePolicyWrapper(base_policy)