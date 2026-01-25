"""
Eval for ZO optimization checkpoints
-------------------------
This script evaluates the top-1 noise from ZO optimization checkpoints (1k/5k/10k)
across all seed folders and computes aggregate statistics.
"""

import argparse
import json
import os
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
from noise_opt_rm.eval.eval_utils import evaluate_noise_single, load_noise_idx, save_eval_serial

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

CHECKPOINT_MAPPING = {
	"1k": "checkpoint_1k_evals",
	"5k": "checkpoint_5k_evals",
	"10k": "checkpoint_10k_evals",
}

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="lift", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/zo_eval_results/")
	p.add_argument("--noise_path", type=str, required=True, help="Path to ZO results directory containing seed folders")
	p.add_argument("--checkpoint", type=str, default=None, choices=["1k", "5k", "10k", None], 
				   help="Checkpoint to evaluate (1k, 5k, 10k, or None for all three)")
	p.add_argument("--save_vid", type=int, default=False)
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	return p.parse_args()

def find_checkpoint_folder(seed_folder_path: str, checkpoint_prefix: str) -> str:
	"""Find the checkpoint folder that starts with the given prefix."""
	folders = [f for f in os.listdir(seed_folder_path) 
			   if f.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(seed_folder_path, f))]
	if not folders:
		raise ValueError(f"No checkpoint folder found with prefix {checkpoint_prefix} in {seed_folder_path}")
	if len(folders) > 1:
		print(f"Warning: Multiple checkpoint folders found with prefix {checkpoint_prefix}, using first one: {folders[0]}")
	return os.path.join(seed_folder_path, folders[0])

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
	
	zo_run_name = os.path.basename(args.noise_path.rstrip('/'))
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Determine which checkpoints to evaluate
	if args.checkpoint is None:
		checkpoints_to_eval = ["1k", "5k", "10k"]
		print(f"Evaluating all checkpoints: {checkpoints_to_eval}")
	else:
		checkpoints_to_eval = [args.checkpoint]
		print(f"Evaluating single checkpoint: {args.checkpoint}")
	
	# Main output directory (one level up if evaluating all checkpoints)
	if args.checkpoint is None:
		main_out = os.path.join(args.out, args.task_name, zo_run_name, 
							   f"eval_all_checkpoints_evalseed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	else:
		main_out = os.path.join(args.out, args.task_name, zo_run_name, 
							   f"eval_{args.checkpoint}_evalseed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	
	os.makedirs(main_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(main_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(main_out, "raw_videos")
	save_vid = args.save_vid
	
	# Find all seed folders
	seed_folders = sorted([d for d in os.listdir(args.noise_path) 
						  if d.startswith('seed_') and os.path.isdir(os.path.join(args.noise_path, d))])
	if not seed_folders:
		raise ValueError(f"No seed folders found in {args.noise_path}")
	
	print(f"\nFound {len(seed_folders)} seed folders: {seed_folders}\n")
	
	# Track results across all checkpoints
	all_checkpoint_results = {}
	
	# Loop over all checkpoints
	for checkpoint_key in checkpoints_to_eval:
		checkpoint_prefix = CHECKPOINT_MAPPING[checkpoint_key]
		
		print(f"\n{'#'*80}")
		print(f"# EVALUATING CHECKPOINT: {checkpoint_key} ({checkpoint_prefix})")
		print(f"# Checkpoint {checkpoints_to_eval.index(checkpoint_key) + 1}/{len(checkpoints_to_eval)}")
		print(f"{'#'*80}\n")
		
		base_out = os.path.join(main_out, f"checkpoint_{checkpoint_key}")
		os.makedirs(base_out, exist_ok=True)
	
		# Track results across all seed folders for this checkpoint
		all_seed_results = {}
	
		# Loop over all seed folders
		for seed_folder in seed_folders:
			zo_seed = seed_folder.split('_')[1]  # Extract seed number from folder name
			print(f"\n{'='*80}")
			print(f"Evaluating seed folder: {seed_folder} ({seed_folders.index(seed_folder) + 1}/{len(seed_folders)})")
			print(f"{'='*80}")
			
			# Find checkpoint folder
			seed_folder_path = os.path.join(args.noise_path, seed_folder)
			checkpoint_path = find_checkpoint_folder(seed_folder_path, checkpoint_prefix)
			
			random.seed(cfg.seed)
			np.random.seed(cfg.seed)
			torch.manual_seed(cfg.seed)
			
			# Load top-1 noise (idx=0)
			best_noise = load_noise_idx(checkpoint_path, eval_idx=0)
			print(f">>> Loaded top-1 noise (idx=0) from {checkpoint_path}")
			
			# Create subdirectory for this seed
			seed_out_dir = os.path.join(base_out, seed_folder)
			os.makedirs(seed_out_dir, exist_ok=True)
			
			# Prepare noise vector once
			noise_vec = best_noise.astype(np.float32).flatten()
			
			rewards_all = []      # List of lists: outer=eval_seeds, inner=evals per seed
			success_flags_all = []    # List of lists: outer=eval_seeds, inner=evals per seed
			episode_seeds = []    # Track actual seed used for each eval iteration

			# Serial rollouts: iterate over evaluation seeds, then evals per seed
			for seed_idx in range(args.n_seeds):
				current_seed = args.seed + seed_idx
				episode_seeds.append(current_seed)
				print(f"\n=== Eval Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
				
				# Build new environment for this seed
				env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
				
				# Collect results for this seed across n_evals_per_seed
				seed_rewards = []
				seed_successes = []
				
				for eval_iter in range(args.n_evals_per_seed):
					print(f"  Evaluation {eval_iter + 1}/{args.n_evals_per_seed}")
					episode_reward, success = evaluate_noise_single(
						env, noise_vec, save_vid, noise_idx=0, 
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
				seed_out_dir, rewards_all, success_flags_all, episode_seeds, 
				ticket_name=f"{zo_run_name}/{seed_folder}",
				eval_noise_idx=0,
				eval_seed=args.seed
			)
			print(
				f"{seed_folder} complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
				f"Success mean={success_mean:.4f} std={success_std:.4f}"
			)
			
			# Store results for this seed folder
			all_seed_results[seed_folder] = {
				"zo_seed": zo_seed,
				"checkpoint_path": checkpoint_path,
				"noise_samples_path": os.path.join(checkpoint_path, "noise_samples.npy"),
				"reward_mean": float(reward_mean),
				"reward_std": float(reward_std),
				"success_mean": float(success_mean),
				"success_std": float(success_std)
			}
		
		# Compute aggregate statistics across all seed folders for this checkpoint
		reward_means = [all_seed_results[sf]["reward_mean"] for sf in seed_folders if sf in all_seed_results]
		success_means = [all_seed_results[sf]["success_mean"] for sf in seed_folders if sf in all_seed_results]
		
		aggregate_stats = {
			"checkpoint": checkpoint_key,
			"checkpoint_prefix": checkpoint_prefix,
			"n_seed_folders": len(all_seed_results),
			"seed_folders": list(all_seed_results.keys()),
			"noise_samples_paths": {sf: all_seed_results[sf]["noise_samples_path"] for sf in all_seed_results},
			"reward_mean_across_seeds": float(np.mean(reward_means)),
			"reward_median_across_seeds": float(np.median(reward_means)),
			"reward_std_across_seeds": float(np.std(reward_means, ddof=1 if len(reward_means) > 1 else 0)),
			"success_mean_across_seeds": float(np.mean(success_means)),
			"success_median_across_seeds": float(np.median(success_means)),
			"success_std_across_seeds": float(np.std(success_means, ddof=1 if len(success_means) > 1 else 0)),
			"per_seed_results": all_seed_results
		}
		
		# Save aggregate results for this checkpoint
		with open(os.path.join(base_out, "aggregate_results.json"), "w") as f:
			json.dump(aggregate_stats, f, indent=2)
		
		print(f"\n{'='*80}")
		print(f"CHECKPOINT {checkpoint_key} AGGREGATE RESULTS")
		print(f"{'='*80}")
		print(f"Number of seed folders evaluated: {len(all_seed_results)}")
		print(f"Reward mean: {aggregate_stats['reward_mean_across_seeds']:.4f} ± {aggregate_stats['reward_std_across_seeds']:.4f}")
		print(f"Reward median: {aggregate_stats['reward_median_across_seeds']:.4f}")
		print(f"Success mean: {aggregate_stats['success_mean_across_seeds']:.4f} ± {aggregate_stats['success_std_across_seeds']:.4f}")
		print(f"Success median: {aggregate_stats['success_median_across_seeds']:.4f}")
		print(f"{'='*80}")
		
		# Store checkpoint results
		all_checkpoint_results[checkpoint_key] = aggregate_stats
	
	# If evaluating all checkpoints, create a summary
	if len(checkpoints_to_eval) > 1:
		summary = {
			"n_checkpoints": len(checkpoints_to_eval),
			"checkpoints": checkpoints_to_eval,
			"comparison": {
				ckpt: {
					"reward_mean": all_checkpoint_results[ckpt]["reward_mean_across_seeds"],
					"reward_std": all_checkpoint_results[ckpt]["reward_std_across_seeds"],
					"success_mean": all_checkpoint_results[ckpt]["success_mean_across_seeds"],
					"success_std": all_checkpoint_results[ckpt]["success_std_across_seeds"]
				}
				for ckpt in checkpoints_to_eval
			},
			"detailed_results": all_checkpoint_results
		}
		
		with open(os.path.join(main_out, "all_checkpoints_summary.json"), "w") as f:
			json.dump(summary, f, indent=2)
		
		print(f"\n{'#'*80}")
		print("SUMMARY ACROSS ALL CHECKPOINTS")
		print(f"{'#'*80}")
		for ckpt in checkpoints_to_eval:
			stats = all_checkpoint_results[ckpt]
			print(f"\n{ckpt}:")
			print(f"  Reward: {stats['reward_mean_across_seeds']:.4f} ± {stats['reward_std_across_seeds']:.4f}")
			print(f"  Success: {stats['success_mean_across_seeds']:.4f} ± {stats['success_std_across_seeds']:.4f}")
		print(f"\nResults saved to: {main_out}/all_checkpoints_summary.json")
		print(f"{'#'*80}")
	else:
		print(f"\nResults saved to: {main_out}")
		
if __name__ == "__main__":
	main()
