"""Eval for lottery tickets found for sub-tasks"""

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

from noise_opt_rm import load_base_policy, build_single_env_with_reward_shaping
from noise_opt_rm.eval.eval_utils import save_eval_serial

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

STAGE_THRESHOLDS = {
	"square": {1: -1.9, 2: -1},
	# Add other tasks as needed
}

def load_tickets_from_subdir(ticket_subdir, st1_idx=0, st2_idx=0):
	"""Load both stage 1 and stage 2 tickets from subdirectory."""
	st1_path = os.path.join(ticket_subdir, "st1_noise_samples.npy")
	st2_path = os.path.join(ticket_subdir, "st2_noise_samples.npy")
	
	st1_samples = np.load(st1_path)
	st2_samples = np.load(st2_path)
	return st1_samples[st1_idx], st2_samples[st2_idx]

def evaluate_multistage_noise(env, st1_noise_vec, st2_noise_vec, reward_thresholds, save_vid=False, noise_idx=0, eval_num=0):
	"""Evaluate multi-stage noise vectors for one episode, switching based on reward thresholds.
	
	Args:
		env: Single environment (not vectorized)
		st1_noise_vec: Stage 1 noise vector
		st2_noise_vec: Stage 2 noise vector
		reward_thresholds: Dict with keys 1 and 2 for stage transition thresholds
		save_vid: Whether to save video
		noise_idx: Index for video naming
		eval_num: Evaluation iteration number
	
	Returns:
		episode_reward: Total episode reward
		success: Whether episode reached success threshold
		st1_success: Whether stage 1 threshold was reached
	"""
	if save_vid:
		env.env.name_prefix = f"st2idx_{noise_idx}_eval_{eval_num}"
	
	env.reset()
	
	# Storage for single episode
	episode_reward = 0.0
	success = False
	st1_success = False
	done = False
	
	# Start with stage 1 noise
	action = st1_noise_vec.reshape(1, -1)
	st2_action = st2_noise_vec.reshape(1, -1)
	
	steps = 0
	while not done:
		_, r, d, info = env.step(action)
		steps += 1
		r_val = float(r[0])  # Extract scalar from array
		episode_reward += r_val
		
		# Check for stage 1 success and switch to stage 2
		if r_val > reward_thresholds[1] and not st1_success:
			action = st2_action  # Switch to stage 2 noise
			st1_success = True
		
		# Check for final success
		if r_val > reward_thresholds[2]:
			success = True 
			done = True  # Terminate early on success
		
		if d[0]:  # Episode terminated or truncated
			done = True
			
	return episode_reward, success, st1_success

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="square", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/noise_eval_results/")
	p.add_argument("--noise_path", type=str, default=None, help="Path to parent directory containing st1_ticket_X subdirectories")
	p.add_argument("--st1_idx", type=int, default=0, help="Stage 1 ticket index to evaluate")
	p.add_argument("--st2_idx", type=int, nargs='+', default=[0], help="List of stage 2 ticket indices to evaluate")
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
	
	# Both stage 1 and stage 2 tickets are in st1_ticket_{st1_idx} subdirectory
	ticket_subdir = os.path.join(args.noise_path, f"st1_ticket_{args.st1_idx}")
	reward_thresholds = STAGE_THRESHOLDS[args.task_name]
	# Extract ticket name from parent directory
	ticket_name = os.path.basename(args.noise_path.rstrip('/')) if args.noise_path else None
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	idx_str = f"st1_{args.st1_idx}_st2_{min(args.st2_idx)}-{max(args.st2_idx)}_n{len(args.st2_idx)}"
	base_out = os.path.join(args.out, args.task_name, ticket_name, f"eval_{idx_str}_seed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	
	os.makedirs(base_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(base_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(base_out, "raw_videos")
	save_vid = args.save_vid
	
	# Track results across all eval indices
	all_indices_results = {}
	
	# Loop over all stage 2 ticket indices
	for st2_idx in args.st2_idx:
		print(f"\n{'='*80}")
		print(f"Stage 1 index: {args.st1_idx} => Evaluating stage 2 noise index: {st2_idx} ({args.st2_idx.index(st2_idx) + 1}/{len(args.st2_idx)})")
		print(f"{'='*80}")
		
		random.seed(cfg.seed)
		np.random.seed(cfg.seed)
		torch.manual_seed(cfg.seed)	

		# Load both tickets
		st1_noise, st2_noise = load_tickets_from_subdir(ticket_subdir, st1_idx=0, st2_idx=st2_idx)
		print(f">>> Loaded stage 1 ticket [{args.st1_idx}] and stage 2 ticket [{st2_idx}] from {ticket_subdir}")
		
		# Create subdirectory for this st2_idx
		args.out = os.path.join(base_out, f"st2_idx_{st2_idx}")
		os.makedirs(args.out, exist_ok=True)
		
		# Prepare noise vectors
		st1_noise_vec = st1_noise.astype(np.float32).flatten()
		st2_noise_vec = st2_noise.astype(np.float32).flatten()
		
		rewards_all = []      # List of lists: outer=seeds, inner=evals per seed
		success_flags_all = []    # List of lists: outer=seeds, inner=evals per seed
		st1_success_all = []  # Track stage 1 success across seeds
		episode_seeds = []    # Track actual seed used for each seed iteration

		# Serial rollouts: iterate over seeds, then evals per seed
		for seed_idx in range(args.n_seeds):
			current_seed = args.seed + seed_idx
			episode_seeds.append(current_seed)
			print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
			
			# Build environment with reward shaping
			env = build_single_env_with_reward_shaping(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
			
			# # Clip noise vectors to action space
			# st1_noise_clipped = np.clip(st1_noise_vec, env.action_space.low, env.action_space.high)
			# st2_noise_clipped = np.clip(st2_noise_vec, env.action_space.low, env.action_space.high)
			
			# Collect results for this seed across n_evals_per_seed
			seed_rewards = []
			seed_successes = []
			seed_st1_successes = []
			
			for eval_iter in range(args.n_evals_per_seed):
				print(f"  Evaluation {eval_iter + 1}/{args.n_evals_per_seed}")
				episode_reward, success, st1_success = evaluate_multistage_noise(
					env, st1_noise_vec, st2_noise_vec, reward_thresholds,
					save_vid, noise_idx=st2_idx, eval_num=eval_iter
				)
				seed_rewards.append(float(episode_reward))
				seed_successes.append(bool(success))
				seed_st1_successes.append(bool(st1_success))
				print(f"    Reward: {episode_reward:.4f}, Success: {success}, Stage1: {st1_success}")
			
			rewards_all.append(seed_rewards)
			success_flags_all.append(seed_successes)
			st1_success_all.append(seed_st1_successes)
			env.close()
			print(f"  Seed {current_seed} complete - Mean reward: {np.mean(seed_rewards):.4f}, Success rate: {np.mean(seed_successes):.4f}, Stage1 rate: {np.mean(seed_st1_successes):.4f}")

		# Save results including stage 1 success
		reward_mean, reward_std, success_mean, success_std = save_eval_serial(
			args.out, rewards_all, success_flags_all, episode_seeds, 
			ticket_name=ticket_name,
			eval_noise_idx=st2_idx,
			eval_seed=args.seed,
			st1_idx=args.st1_idx,
			st2_idx=st2_idx
		)
		
		# Compute stage 1 success statistics
		st1_success_matrix = np.array(st1_success_all, dtype=bool)
		st1_seed_means = st1_success_matrix.mean(axis=1) if st1_success_matrix.size else np.array([])
		st1_success_mean = float(st1_seed_means.mean()) if st1_seed_means.size else 0.0
		st1_success_std = float(st1_seed_means.std(ddof=1)) if st1_seed_means.size else 0.0
		
		print(
			f"Stage 2 noise {st2_idx} complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
			f"Success mean={success_mean:.4f} std={success_std:.4f}; "
			f"Stage1 mean={st1_success_mean:.4f} std={st1_success_std:.4f}"
		)
		
		# Store results for this index
		all_indices_results[st2_idx] = {
			"reward_mean": float(reward_mean),
			"reward_std": float(reward_std),
			"success_mean": float(success_mean),
			"success_std": float(success_std),
			"st1_success_mean": float(st1_success_mean),
			"st1_success_std": float(st1_success_std)
		}
	
	# Compute aggregate statistics across all eval indices
	if len(args.st2_idx) > 1:
		reward_means = [all_indices_results[idx]["reward_mean"] for idx in args.st2_idx]
		success_means = [all_indices_results[idx]["success_mean"] for idx in args.st2_idx]
		st1_success_means = [all_indices_results[idx]["st1_success_mean"] for idx in args.st2_idx]
		
		aggregate_stats = {
			"n_st2_indices": len(args.st2_idx),
			"st1_idx": args.st1_idx,
			"st2_indices": args.st2_idx,
			"reward_thresholds": reward_thresholds,
			"reward_mean_across_indices": float(np.mean(reward_means)),
			"reward_std_across_indices": float(np.std(reward_means, ddof=1 if len(reward_means) > 1 else 0)),
			"success_mean_across_indices": float(np.mean(success_means)),
			"success_std_across_indices": float(np.std(success_means, ddof=1 if len(success_means) > 1 else 0)),
			"st1_success_mean_across_indices": float(np.mean(st1_success_means)),
			"st1_success_std_across_indices": float(np.std(st1_success_means, ddof=1 if len(st1_success_means) > 1 else 0)),
			"per_index_results": all_indices_results
		}
		
		# Save aggregate results
		with open(os.path.join(base_out, "aggregate_results.json"), "w") as f:
			json.dump(aggregate_stats, f, indent=2)
		
		print(f"\n{'='*80}")
		print("AGGREGATE RESULTS ACROSS ALL STAGE 2 NOISE INDICES")
		print(f"{'='*80}")
		print(f"Stage 1 index: {args.st1_idx}")
		print(f"Number of stage 2 indices evaluated: {len(args.st2_idx)}")
		print(f"Reward mean across indices: {aggregate_stats['reward_mean_across_indices']:.4f} ± {aggregate_stats['reward_std_across_indices']:.4f}")
		print(f"Success mean across indices: {aggregate_stats['success_mean_across_indices']:.4f} ± {aggregate_stats['success_std_across_indices']:.4f}")
		print(f"Stage 1 success mean across indices: {aggregate_stats['st1_success_mean_across_indices']:.4f} ± {aggregate_stats['st1_success_std_across_indices']:.4f}")
		print(f"Results saved to: {base_out}/aggregate_results.json")
		print(f"{'='*80}")

if __name__ == "__main__":
	main()
