# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"

BASE_DIR = Path(__file__).resolve().parents[2]
if BASE_DIR.as_posix() not in sys.path:
	sys.path.append(BASE_DIR.as_posix())

from env_util import build_single_env
from policy_util import load_base_policy
from eval_utils import save_eval_serial

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def parse_args() -> argparse.Namespace:
	"""Helper function to parse arguments."""
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5)
	p.add_argument("--seed", type=int, default=1619, help="Starting seed")
	p.add_argument("--out", default="logs_res_rm/policy_eval/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--save_vid", action="store_true", help="Save evaluation videos")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_seeds: int, n_evals_per_seed: int, seed: int, ddim_steps: int) -> str:
	ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	run_name = f"base_policy_seeds{n_seeds}_evals{n_evals_per_seed}_seed{seed}_ddim{ddim_steps}_{ts}"
	return os.path.join(out_path.rstrip("/"), task_name, run_name)

def evaluate_policy_single(env, save_vid=False, eval_num=0, rew_offset=0.0):
	if save_vid:
		env.env.name_prefix = f"ep_{eval_num}"
	
	env.reset()
	episode_reward = 0.0
	success = False
	done = False

	action_space_dim = env.action_space.shape[0]
	action = np.zeros((1, action_space_dim), dtype=np.float32) # controllability noise disabled

	steps = 0
	while not done:
		_, r, d, info = env.step(action)
		steps += 1
		r_val = float(r[0])
		episode_reward += r_val
		
		if r_val > float(-rew_offset):
			success = True
		
		if d[0]:
			done = True
			
	return episode_reward, success

def main():
	args = parse_args()
	base_path = BASE_DIR.as_posix()
	config_path = os.path.join(f"{base_path}/lottery_tickets/robomimic_dppo_lt", TASK_CONFIGS[args.task_name])

	OmegaConf.register_new_resolver("eval", eval)
	config_dir = os.path.dirname(config_path)
	config_name = os.path.basename(config_path).replace(".yaml", "")
	with initialize_config_dir(version_base=None, config_dir=config_dir):
		cfg = compose(config_name=config_name)
	OmegaConf.set_struct(cfg, False)
	cfg.seed = args.seed
	if not hasattr(cfg, "device") or cfg.device is None:
		cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
	
	# Override ddim_steps if provided
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	
	args.out = _resolve_out(args.out, args.task_name, args.n_seeds, args.n_evals_per_seed, args.seed, cfg.model.ddim_steps)
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	cfg.model.controllable_noise = False # IMP
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = args.save_vid
	
	rewards_all = []
	success_flags_all = []
	episode_seeds = []

	for seed_idx in range(args.n_seeds):
		current_seed = args.seed + seed_idx
		episode_seeds.append(current_seed)
		print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
		
		env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
		
		seed_rewards = []
		seed_successes = []
		
		for eval_idx in range(args.n_evals_per_seed):
			print(f"  Evaluation {eval_idx + 1}/{args.n_evals_per_seed}")
			episode_reward, success = evaluate_policy_single(
				env, save_vid, eval_num=eval_idx, rew_offset=cfg.env.reward_offset
			)
			seed_rewards.append(float(episode_reward))
			seed_successes.append(bool(success))
			print(f"    Reward: {episode_reward:.4f}, Success: {success}")
		
		rewards_all.append(seed_rewards)
		success_flags_all.append(seed_successes)
		env.close()
		print(f"  Seed {current_seed} complete - Mean reward: {np.mean(seed_rewards):.4f}, Success rate: {np.mean(seed_successes):.4f}")

	reward_mean, reward_std, success_mean, success_std = save_eval_serial(
		args.out, rewards_all, success_flags_all, episode_seeds,
		eval_type="base_policy",
		eval_seed=args.seed,
		n_seeds=args.n_seeds,
		n_evals_per_seed=args.n_evals_per_seed
	)
	print(
		f"Evaluation complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
		f"Success mean={success_mean:.4f} std={success_std:.4f}"
	)

if __name__ == "__main__":
	main()
