"""Eval for budegt opt results from lottery ticket search"""

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

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/noise_eval_results/")
	p.add_argument("--ticket_path", type=str, required=True, help="Path to lottery ticket results directory containing seed folders")
	p.add_argument("--checkpoint", type=int, default=499, help="Checkpoint number to evaluate (e.g., 499 for checkpoint_499)")
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
	
	ticket_name = os.path.basename(args.ticket_path.rstrip('/'))
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	checkpoint_name = f"checkpoint_{args.checkpoint}"
	base_out = os.path.join(args.out, args.task_name, ticket_name, f"eval_{checkpoint_name}_evalseed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	
	os.makedirs(base_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(base_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(base_out, "raw_videos")
	save_vid = args.save_vid
	
	# Find all seed folders and evaluate best noise from checkpoint
	seed_folders = sorted([d for d in os.listdir(args.ticket_path) if d.startswith('seed_') and os.path.isdir(os.path.join(args.ticket_path, d))])
	if not seed_folders:
		raise ValueError(f"No seed folders found in {args.ticket_path}")
	
	print(f"\nFound {len(seed_folders)} seed folders: {seed_folders}")
	print(f"Evaluating best noise (idx=0) from checkpoint: {checkpoint_name}\n")
	
	# Track results across all seed folders
	all_seed_results = {}
	
	# Loop over all seed folders
	for seed_folder in seed_folders:
		ticket_seed = seed_folder.split('_')[1]  # Extract seed number from folder name
		print(f"\n{'='*80}")
		print(f"Evaluating seed folder: {seed_folder} ({seed_folders.index(seed_folder) + 1}/{len(seed_folders)})")
		print(f"{'='*80}")
		
		# Load best noise (idx=0) from this seed's checkpoint
		checkpoint_path = os.path.join(args.ticket_path, seed_folder, checkpoint_name)
		
		random.seed(cfg.seed)
		np.random.seed(cfg.seed)
		torch.manual_seed(cfg.seed)
		
		best_noise = load_noise_idx(checkpoint_path, eval_idx=0)  # Always use idx=0 (best)
		print(f">>> Loaded best noise (idx=0) from {checkpoint_path}")
		
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
			# noise_vec_clipped = np.clip(noise_vec, env.action_space.low, env.action_space.high)
			
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
			ticket_name=f"{ticket_name}/{seed_folder}",
			eval_noise_idx=0,
			eval_seed=args.seed
		)
		print(
			f"{seed_folder} complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
			f"Success mean={success_mean:.4f} std={success_std:.4f}"
		)
		
		# Store results for this seed folder
		all_seed_results[seed_folder] = {
			"ticket_seed": ticket_seed,
			"reward_mean": float(reward_mean),
			"reward_std": float(reward_std),
			"success_mean": float(success_mean),
			"success_std": float(success_std)
		}
	
	# Compute aggregate statistics across all seed folders
	reward_means = [all_seed_results[sf]["reward_mean"] for sf in seed_folders if sf in all_seed_results]
	success_means = [all_seed_results[sf]["success_mean"] for sf in seed_folders if sf in all_seed_results]
	
	aggregate_stats = {
		"checkpoint": checkpoint_name,
		"n_seed_folders": len(all_seed_results),
		"seed_folders": list(all_seed_results.keys()),
		"reward_mean_across_seeds": float(np.mean(reward_means)),
		"reward_std_across_seeds": float(np.std(reward_means, ddof=1 if len(reward_means) > 1 else 0)),
		"success_mean_across_seeds": float(np.mean(success_means)),
		"success_std_across_seeds": float(np.std(success_means, ddof=1 if len(success_means) > 1 else 0)),
		"per_seed_results": all_seed_results
	}
	
	# Save aggregate results
	with open(os.path.join(base_out, "aggregate_results.json"), "w") as f:
		json.dump(aggregate_stats, f, indent=2)
	
	print(f"\n{'='*80}")
	print("AGGREGATE RESULTS ACROSS ALL SEED FOLDERS")
	print(f"{'='*80}")
	print(f"Checkpoint: {checkpoint_name}")
	print(f"Number of seed folders evaluated: {len(all_seed_results)}")
	print(f"Reward mean across seeds: {aggregate_stats['reward_mean_across_seeds']:.4f} ± {aggregate_stats['reward_std_across_seeds']:.4f}")
	print(f"Success mean across seeds: {aggregate_stats['success_mean_across_seeds']:.4f} ± {aggregate_stats['success_std_across_seeds']:.4f}")
	print(f"Results saved to: {base_out}/aggregate_results.json")
	print(f"{'='*80}")

if __name__ == "__main__":
	main()
