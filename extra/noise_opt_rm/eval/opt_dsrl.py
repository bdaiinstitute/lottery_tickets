"""Eval for DSRL SAC and DSRL-NA policies on Robomimic tasks."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import glob

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
from noise_opt_rm.eval.eval_utils import save_eval_serial
from stable_baselines3 import SAC, DSRL

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def find_seed_dirs(parent_dir, algorithm):
	"""Find all seed directories matching the algorithm in parent directory."""
	pattern = f"*_dsrl_{algorithm.replace('dsrl_', '')}_*"
	seed_dirs = sorted([d for d in glob.glob(os.path.join(parent_dir, pattern)) if os.path.isdir(d)])
	return seed_dirs

def load_checkpoint(checkpoint_dir, episodes, algorithm):
	"""Load SAC/DSRL checkpoint from directory."""
	# Find the timestamped subdirectory within checkpoint_dir
	# Then look for checkpoint files in timestamped_dir/checkpoint/
	subdirs = [d for d in glob.glob(os.path.join(checkpoint_dir, "*")) if os.path.isdir(d)]
	if not subdirs:
		raise FileNotFoundError(f"No subdirectories found in {checkpoint_dir}")
	
	# Use the first (and typically only) timestamped subdirectory
	timestamped_dir = subdirs[0]
	checkpoint_subdir = os.path.join(timestamped_dir, "checkpoint")
	pattern = os.path.join(checkpoint_subdir, f"ft_policy_{episodes}ep_*steps.zip")
	checkpoint_files = glob.glob(pattern)
	
	if not checkpoint_files:
		raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
	
	if len(checkpoint_files) > 1:
		print(f"Warning: Multiple checkpoints found for {episodes} episodes, using first: {checkpoint_files[0]}")
	
	checkpoint_path = checkpoint_files[0]
	print(f"Loading checkpoint: {checkpoint_path}")
	
	# Load based on algorithm type
	if algorithm == 'dsrl_na':
		model = DSRL.load(checkpoint_path)
		print("Loaded as DSRL model")
	elif algorithm == 'dsrl_sac':
		model = SAC.load(checkpoint_path)
		print("Loaded as SAC model")
	else:
		raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dsrl_na' or 'dsrl_sac'")
	return model

def evaluate_policy_single(env, policy, save_vid=False, episodes=0, eval_num=0, rew_offset=1.0, deterministic=True):
	"""Evaluate policy for one episode using policy.predict()."""
	if save_vid:
		env.env.name_prefix = f"episodes_{episodes}_eval_{eval_num}"
	
	obs = env.reset()
	episode_reward = 0.0
	success = False
	done = False
	
	while not done:
		# Get action from policy
		action, _ = policy.predict(obs, deterministic=deterministic)
		
		# Step environment
		obs, r, d, info = env.step(action)
		r_val = float(r[0])  # Extract scalar from array
		episode_reward += r_val
		
		if r_val > float(-rew_offset):  # terminated successfully
			success = True
		
		if d[0]:  # truncated or terminated
			done = True
			
	return episode_reward, success

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/dsrl_eval_results/")
	p.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory or parent directory containing multiple seed runs")
	p.add_argument("--algorithm", type=str, required=True, choices=['dsrl_sac', 'dsrl_na'], help="Algorithm type: 'dsrl_sac' or 'dsrl_na'")
	p.add_argument("--eval_episodes", type=int, nargs='+', default=[1000, 5000, 10000], help="List of episode checkpoints to evaluate")
	p.add_argument("--save_vid", type=int, default=False)
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--eval_across_seeds", action="store_true", help="Evaluate across multiple seed directories in parent dir")
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
	
	# Determine checkpoint directories to evaluate
	if args.eval_across_seeds:
		checkpoint_dirs = find_seed_dirs(args.checkpoint_dir, args.algorithm)
		if not checkpoint_dirs:
			raise ValueError(f"No seed directories found in {args.checkpoint_dir} for algorithm {args.algorithm}")
		print(f"Found {len(checkpoint_dirs)} seed directories to evaluate")
		run_name = os.path.basename(args.checkpoint_dir)
	else:
		checkpoint_dirs = [args.checkpoint_dir]
		run_name = os.path.basename(args.checkpoint_dir)
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	episodes_str = f"ep{'_'.join(map(str, args.eval_episodes))}"
	suffix = "across_seeds" if args.eval_across_seeds else f"seed{args.seed}"
	algo_suffix = args.algorithm.replace('dsrl_', '')
	base_out = os.path.join(args.out, args.task_name, run_name, f"eval_{episodes_str}_{suffix}_{algo_suffix}_ddim{cfg.model.ddim_steps}_{timestamp}")
	
	os.makedirs(base_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(base_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(base_out, "raw_videos")
	save_vid = args.save_vid
	
	# Track results across all episode checkpoints and seeds
	all_episodes_results = {}
	
	# Loop over all episode checkpoints
	for episodes in args.eval_episodes:
		print(f"\n{'='*80}")
		print(f"Evaluating checkpoint: {episodes} episodes ({args.eval_episodes.index(episodes) + 1}/{len(args.eval_episodes)})")
		print(f"{'='*80}")
		
		# Track results across all seed directories for this checkpoint
		seed_dir_results = {"rewards": [], "successes": []}
		
		# Loop over all seed directories
		for seed_dir_idx, checkpoint_dir in enumerate(checkpoint_dirs):
			print(f"\n--- Seed directory {seed_dir_idx + 1}/{len(checkpoint_dirs)}: {os.path.basename(checkpoint_dir)} ---")
			
			random.seed(cfg.seed)
			np.random.seed(cfg.seed)
			torch.manual_seed(cfg.seed)

			# Load checkpoint for this episode count and seed directory
			rl_policy = load_checkpoint(checkpoint_dir, episodes, args.algorithm)
			
			# Create subdirectory for this episode checkpoint and seed directory
			seed_dir_name = os.path.basename(checkpoint_dir)
			episodes_out = os.path.join(base_out, f"episodes_{episodes}", seed_dir_name)
			episodes_out = os.path.join(base_out, f"episodes_{episodes}", seed_dir_name)
			os.makedirs(episodes_out, exist_ok=True)
			
			rewards_all = []      # List of lists: outer=seeds, inner=evals per seed
			success_flags_all = []    # List of lists: outer=seeds, inner=evals per seed
			episode_seeds = []    # Track actual seed used for each seed iteration

			# Serial rollouts: iterate over seeds, then evals per seed
			for seed_idx in range(args.n_seeds):
				current_seed = args.seed + seed_idx
				episode_seeds.append(current_seed)
				print(f"\n=== Eval seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
				
				# Build new environment for this seed
				env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
				
				# Collect results for this seed across n_evals_per_seed
				seed_rewards = []
				seed_successes = []
				
				for eval_iter in range(args.n_evals_per_seed):
					print(f"  Evaluation {eval_iter + 1}/{args.n_evals_per_seed}")
					episode_reward, success = evaluate_policy_single(
						env, rl_policy, save_vid=save_vid, episodes=episodes, 
						eval_num=eval_iter, rew_offset=cfg.env.reward_offset, deterministic=True
					)
					seed_rewards.append(float(episode_reward))
					seed_successes.append(bool(success))
					print(f"    Reward: {episode_reward:.4f}, Success: {success}")
				
				rewards_all.append(seed_rewards)
				success_flags_all.append(seed_successes)
				env.close()
				print(f"  Eval seed {current_seed} complete - Mean reward: {np.mean(seed_rewards):.4f}, Success rate: {np.mean(seed_successes):.4f}")

			reward_mean, reward_std, success_mean, success_std = save_eval_serial(
				episodes_out, rewards_all, success_flags_all, episode_seeds, 
				run_name=seed_dir_name,
				eval_episodes=episodes,
				eval_seed=args.seed
			)
			print(
				f"Seed dir {seed_dir_name} complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
				f"Success mean={success_mean:.4f} std={success_std:.4f}"
			)
			
			# Store results for this seed directory
			seed_dir_results["rewards"].append(float(reward_mean))
			seed_dir_results["successes"].append(float(success_mean))
		
		# Compute statistics across seed directories for this checkpoint
		reward_mean_across_seeds = float(np.mean(seed_dir_results["rewards"]))
		reward_std_across_seeds = float(np.std(seed_dir_results["rewards"], ddof=1 if len(seed_dir_results["rewards"]) > 1 else 0))
		success_mean_across_seeds = float(np.mean(seed_dir_results["successes"]))
		success_std_across_seeds = float(np.std(seed_dir_results["successes"], ddof=1 if len(seed_dir_results["successes"]) > 1 else 0))
		
		print(f"\nEpisodes {episodes} across {len(checkpoint_dirs)} seeds complete.")
		print(f"Reward mean={reward_mean_across_seeds:.4f} std={reward_std_across_seeds:.4f}; "
		      f"Success mean={success_mean_across_seeds:.4f} std={success_std_across_seeds:.4f}")
		
		# Store results for this episode checkpoint
		all_episodes_results[episodes] = {
			"reward_mean": reward_mean_across_seeds,
			"reward_std": reward_std_across_seeds,
			"success_mean": success_mean_across_seeds,
			"success_std": success_std_across_seeds,
			"n_seed_dirs": len(checkpoint_dirs),
			"per_seed_rewards": seed_dir_results["rewards"],
			"per_seed_successes": seed_dir_results["successes"]
		}
	
	# Save all episode results
	with open(os.path.join(base_out, "all_episodes_results.json"), "w") as f:
		json.dump(all_episodes_results, f, indent=2)
	
	print(f"\nAll evaluations complete. Results saved to: {base_out}/all_episodes_results.json")

if __name__ == "__main__":
	main()
