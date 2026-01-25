"""Eval for Gaussian-sampled noise."""

import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["MUJOCO_GL"] = "egl"

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_single_env, load_base_policy
from noise_opt_rm.eval.eval_utils import evaluate_gaussian_single, save_eval_serial

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/gauss_noise_eval_results/")
	p.add_argument("--gaussian_model", type=str, required=True, help="Path to fitted Gaussian model (.pkl)")
	p.add_argument("--save_vid", type=int, default=False)
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	return p.parse_args()

def main():
	args = p_args()
	base_path = str(BASE_DIR)
	config_path = os.path.join(base_path, TASK_CONFIGS[args.task_name])
	OmegaConf.register_new_resolver("eval", eval)
	config_dir = os.path.dirname(config_path)
	config_name = os.path.basename(config_path).replace('.yaml', '')
	
	with initialize_config_dir(version_base=None, config_dir=config_dir):
		cfg = compose(config_name=config_name)
	OmegaConf.set_struct(cfg, False)
	cfg.seed = args.seed
	if not hasattr(cfg, 'device') or cfg.device is None:
		cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	
	# Override ddim_steps if provided
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps
	
	# Extract Gaussian model name
	gauss_name = os.path.basename(args.gaussian_model).replace('.pkl', '')
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_out = os.path.join(args.out, args.task_name, gauss_name, f"eval_seed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	os.makedirs(base_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(base_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(base_out, "raw_videos")
	save_vid = args.save_vid
	
	# Load Gaussian model
	with open(args.gaussian_model, 'rb') as f:
		gauss_model = pickle.load(f)
	mean = gauss_model['mean']
	cov = gauss_model['cov']
	print(f">>> Loaded Gaussian model from {args.gaussian_model}")
	print(f">>> Mean shape: {mean.shape}, Cov shape: {cov.shape}")
	
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	
	# Create RNG for Gaussian sampling
	rng = np.random.default_rng(cfg.seed)
	
	rewards_all = []      # List of lists: outer=seeds, inner=evals per seed
	success_flags_all = []    # List of lists: outer=seeds, inner=evals per seed
	episode_seeds = []    # Track actual seed used for each seed iteration

	# Serial rollouts: iterate over seeds, then evals per seed
	for seed_idx in range(args.n_seeds):
		current_seed = args.seed + seed_idx
		episode_seeds.append(current_seed)
		print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
		
		# Build new environment for this seed
		env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
		
		# Collect results for this seed across n_evals_per_seed
		seed_rewards = []
		seed_successes = []
		
		for eval_iter in range(args.n_evals_per_seed):
			print(f"  Evaluation {eval_iter + 1}/{args.n_evals_per_seed}")
			
			# Sample from Gaussian at every step (done inside evaluate_gaussian_single)
			episode_reward, success = evaluate_gaussian_single(
				env, mean, cov, rng, save_vid, noise_idx=eval_iter, 
				eval_num=eval_iter, rew_offset=cfg.env.reward_offset
			)
			seed_rewards.append(float(episode_reward))
			seed_successes.append(bool(success))
			print(f"    Reward: {episode_reward:.4f}, Success: {success}")
		
		rewards_all.append(seed_rewards)
		success_flags_all.append(seed_successes)
		env.close()
		print(f"  Seed {current_seed} complete - Mean reward: {np.mean(seed_rewards):.4f}, Success rate: {np.mean(seed_successes):.4f}")

	reward_mean, reward_std, success_mean, success_std = save_eval_serial(
		base_out, rewards_all, success_flags_all, episode_seeds, 
		ticket_name=gauss_name,
		eval_noise_idx=0,
		eval_seed=args.seed
	)
	print(
		f"\nGaussian sampling complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
		f"Success mean={success_mean:.4f} std={success_std:.4f}"
	)

if __name__ == "__main__":
	main()
