"""Zeroth-order search for optimal noise vectors in robotic manipulation tasks.

Algorithm:
1. Start with a random Gaussian noise n as pivot
2. Find N candidates in the pivot's neighborhood S_λ(n) = {y : d(y, n) = λ}
3. Evaluate candidates using rollouts (success rate)
4. Select the best candidate as the new pivot and repeat steps 2-3

Roughly we can afford 10,000 episodes. If we do 10 evals per ticket, that leaves us 1000 tickets to evaluate.
For zeroth order search, we can do 100 iterations with a population of 10 tickets each.
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
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

os.environ["MUJOCO_GL"] = "egl"
# os.environ["WANDB_MODE"] = "offline"

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_lt_env, load_base_policy
from noise_opt_rm.ticket_search.rollout.lottery_ticket import evaluate_noise, save_results
from noise_opt_rm.eval.eval_utils import load_noise_idx

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments for evaluation")
	p.add_argument("--n_iterations", type=int, default=100, help="Number of ZO search iterations")
	p.add_argument("--n_candidates", type=int, default=10, help="Number of candidates per iteration")
	p.add_argument("--lambda_radius", type=float, default=0.5, help="Neighborhood radius lambda")
	p.add_argument("--distance_metric", type=str, default="l2", choices=["l2", "linf"], 
				   help="Distance metric for neighborhood")
	p.add_argument("--seed", type=int, default=666)
	p.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to run")
	p.add_argument("--exp_name", type=str, default="", help="Experiment name to append to output folder")
	p.add_argument("--out", default="logs_res_rm/zo_ticket_results/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--noise_path", type=str, default=None, help="Path to .npy file with initial pivot noise")
	p.add_argument("--eval_idx", type=int, default=0, help="Index of noise sample to load from noise_path")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_envs: int, n_iterations: int, n_candidates: int, 
				 lambda_radius: float, seed: int, ddim_steps: int, exp_name: str = "", 
				 noise_path: str = None, eval_idx: int = None) -> str:
	ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
	run_name = f"zo_envs{n_envs}_iter{n_iterations}_cand{n_candidates}_lambda{lambda_radius}_seed{seed}_ddim{ddim_steps}_{ts}"
	if exp_name:
		run_name = f"{run_name}_{exp_name}"
	
	# Add parent folder if noise_path and eval_idx are specified
	if noise_path is not None and eval_idx is not None:
		# Extract the parent directory name from noise_path
		noise_dir = os.path.basename(os.path.normpath(noise_path))
		parent_folder = f"{noise_dir}_idx{eval_idx}"
		return os.path.join(out_path.rstrip('/'), task_name, parent_folder, run_name)
	
	return os.path.join(out_path.rstrip('/'), task_name, run_name)

def sample_neighborhood(pivot: np.ndarray, lambda_radius: float, n_samples: int, 
					   action_low: np.ndarray, action_high: np.ndarray,
					   distance_metric: str = "l2", rng: np.random.Generator = None,
					   winning_direction: np.ndarray = None) -> np.ndarray:
	"""Sample candidates in the neighborhood of the pivot.
	
	Args:
		pivot: Current pivot noise vector (action_dim,)
		lambda_radius: Radius of the neighborhood
		n_samples: Number of candidates to sample
		action_low: Lower bound of action space
		action_high: Upper bound of action space
		distance_metric: Distance metric ("l2" or "linf")
		rng: Random number generator
		winning_direction: Optional winning direction from previous iteration to include
		
	Returns:
		Array of shape (n_samples, action_dim) containing candidate noise vectors
	"""
	if rng is None:
		rng = np.random.default_rng()
	
	candidates = []
	action_dim = len(pivot)
	
	# Always include the winning direction from the last iteration if provided
	if winning_direction is not None:
		candidate = pivot + winning_direction
		candidates.append(candidate)
		n_samples -= 1  # Reduce the number of random samples by 1
	
	for _ in range(n_samples):
		if distance_metric == "l2":
			# Sample uniformly on the L2 sphere of radius lambda
			direction = rng.standard_normal(action_dim)
			direction = direction / np.linalg.norm(direction)
			candidate = pivot + lambda_radius * direction
		elif distance_metric == "linf":
			# Sample uniformly on the L-infinity sphere of radius lambda
			# This is a hypercube boundary
			direction = rng.standard_normal(action_dim)
			direction = direction / np.abs(direction).max()
			candidate = pivot + lambda_radius * direction
		else:
			raise ValueError(f"Unknown distance metric: {distance_metric}")
		
		candidates.append(candidate)
	
	return np.array(candidates, dtype=np.float32)

def main():
	args = p_args()
	base_path = str(BASE_DIR)
	config_path = os.path.join(base_path, TASK_CONFIGS[args.task_name])
	
	# Register the eval resolver for OmegaConf
	OmegaConf.register_new_resolver("eval", eval)
	
	# Use Hydra to load config with proper interpolation support
	config_dir = os.path.dirname(config_path)
	config_name = os.path.basename(config_path).replace('.yaml', '')
	
	with initialize_config_dir(version_base=None, config_dir=config_dir):
		cfg = compose(config_name=config_name)

	# Allow mutation of config
	OmegaConf.set_struct(cfg, False)
	
	# Override ddim_steps if provided
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps
	
	if not hasattr(cfg, 'device') or cfg.device is None:
		cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	
	# Create outer directory structure once
	base_out = _resolve_out(args.out, args.task_name, args.n_envs, args.n_iterations, 
						   args.n_candidates, args.lambda_radius, args.seed, cfg.model.ddim_steps, args.exp_name,
						   noise_path=args.noise_path, eval_idx=args.eval_idx)
	
	# Initialize wandb once for all seeds
	wandb_run = wandb.init(
		project="zo-noise-search",
		entity=None,
		name=f"{args.task_name}_lambda{args.lambda_radius}_nseeds{args.n_seeds}",
		config={
			"task_name": args.task_name,
			"n_envs": args.n_envs,
			"n_iterations": args.n_iterations,
			"n_candidates": args.n_candidates,
			"lambda_radius": args.lambda_radius,
			"distance_metric": args.distance_metric,
			"base_seed": args.seed,
			"n_seeds": args.n_seeds,
			"ddim_steps": cfg.model.ddim_steps,
			"noise_path": args.noise_path,
			"eval_idx": args.eval_idx,
		}
	)
	
	# Loop over seeds
	for seed_idx in range(args.n_seeds):
		current_seed = args.seed + seed_idx
		print(f"\n{'='*80}")
		print(f"Running seed {seed_idx + 1}/{args.n_seeds}: {current_seed}")
		print(f"{'='*80}\n")
		
		cfg.seed = current_seed
		
		# Create seed-specific subdirectory
		seed_out = os.path.join(base_out, f"seed_{current_seed}")
		
		# Set random seeds
		random.seed(current_seed)
		torch.manual_seed(current_seed)
		np.random.seed(current_seed)
		
		# Load policy and build environment
		base_policy = load_base_policy(cfg)
		video_dir = os.path.join(seed_out, "raw_videos")
		save_vid = args.n_envs < 10
		
		# Derive and fix per-env seeds
		ss_env = np.random.SeedSequence(current_seed)
		fixed_seeds = [int(s.generate_state(1)[0]) for s in ss_env.spawn(args.n_envs)]
		env = build_lt_env(base_policy, cfg, args.n_envs, video_dir, save_vid=save_vid, fixed_seeds=fixed_seeds)
		
		# Create RNG for noise generation
		rng = np.random.default_rng(current_seed)
		
		# Get action space dimension
		action_space_dim = env.action_space.shape[0]
		action_low = env.action_space.low
		action_high = env.action_space.high
		
		# Save initial observation for verification
		initial_obs = env.reset()
		os.makedirs(seed_out, exist_ok=True)
		
		# Dump resolved config
		with open(os.path.join(seed_out, "config.yaml"), "w") as f:
			f.write(OmegaConf.to_yaml(cfg))
		
		np.save(os.path.join(seed_out, "initial_obs.npy"), initial_obs)
		print(f"Saved initial observation with shape {initial_obs.shape}")
		
		# Initialize tracking
		all_noise = []
		all_rewards = []
		all_success = []
		all_success_rates = []
		all_lengths = []
		pivot_history = []
		pivot_success_history = []
		
		# Step 1: Initialize with random Gaussian noise as pivot (or load from file)
		if args.noise_path is not None:
			print(f"Loading initial pivot from {args.noise_path} at index {args.eval_idx}")
			pivot = load_noise_idx(args.noise_path, eval_idx=args.eval_idx).astype(np.float32)
			if pivot.shape[0] != action_space_dim:
				raise ValueError(f"Loaded noise has dimension {pivot.shape[0]}, expected {action_space_dim}")
		else:
			pivot = rng.standard_normal(action_space_dim).astype(np.float32)
		
		# Evaluate initial pivot
		print(f"Evaluating initial pivot...")
		per_env_reward, per_env_success, per_env_length = evaluate_noise(
			env, pivot, args.n_envs, save_vid=False, noise_idx=0, 
			rew_offset=cfg.env.reward_offset, expected_initial_obs=initial_obs
		)
		pivot_success_rate = float(np.mean(per_env_success))
		
		all_noise.append(pivot.copy())
		all_rewards.append(per_env_reward)
		all_success.append(per_env_success)
		all_success_rates.append(pivot_success_rate)
		all_lengths.append(per_env_length)
		pivot_history.append(pivot.copy())
		pivot_success_history.append(pivot_success_rate)
		
		print(f"Initial pivot success rate: {pivot_success_rate:.4f}")
		
		wandb.log({
			"seed": current_seed,
			"iteration": 0,
			"pivot_success_rate": pivot_success_rate,
			"pivot_mean_reward": float(np.mean(per_env_reward)),
			"pivot_mean_length": float(np.mean(per_env_length)),
			"total_evaluations": args.n_envs,
		})
		
		total_samples = 1  # Count initial pivot
		winning_direction = None  # Track the winning direction from previous iteration
		
		# Main ZO search loop
		for iteration in range(args.n_iterations):
			print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")
			print(f"Current pivot success rate: {pivot_success_rate:.4f}")
			
			# Step 2: Sample candidates in the neighborhood
			candidates = sample_neighborhood(
				pivot, args.lambda_radius, args.n_candidates,
				action_low, action_high, args.distance_metric, rng,
				winning_direction=winning_direction
			)
			
			# Step 3: Evaluate all candidates
			best_candidate = None
			best_candidate_success_rate = pivot_success_rate
			best_candidate_mean_reward = float(np.mean(all_rewards[0]))  # Mean reward of initial pivot
			best_candidate_rewards = None
			best_candidate_successes = None
			
			for cand_idx, candidate in enumerate(candidates):
				noise_idx = total_samples + cand_idx
				print(f"Evaluating candidate {cand_idx + 1}/{args.n_candidates} (sample {noise_idx})...")
				
				per_env_reward, per_env_success, per_env_length = evaluate_noise(
					env, candidate, args.n_envs, save_vid=save_vid, noise_idx=noise_idx,
					rew_offset=cfg.env.reward_offset, expected_initial_obs=initial_obs
				)
				success_rate = float(np.mean(per_env_success))
				mean_reward = float(np.mean(per_env_reward))
				
				all_noise.append(candidate.copy())
				all_rewards.append(per_env_reward)
				all_success.append(per_env_success)
				all_success_rates.append(success_rate)
				all_lengths.append(per_env_length)
				
				print(f"  Success rate: {success_rate:.4f}, Mean reward: {mean_reward:.4f}")
				
				wandb.log({
					"seed": current_seed,
					"iteration": iteration + 1,
					"candidate_idx": cand_idx + 1,
					"candidate_success_rate": success_rate,
					"candidate_mean_reward": mean_reward,
					"candidate_mean_length": float(np.mean(per_env_length)),
				})
				
				# Track the best candidate (use mean reward to break ties)
				if (success_rate > best_candidate_success_rate or 
					(success_rate == best_candidate_success_rate and mean_reward > best_candidate_mean_reward)):
					best_candidate = candidate.copy()
					best_candidate_success_rate = success_rate
					best_candidate_mean_reward = mean_reward
					best_candidate_rewards = per_env_reward
					best_candidate_successes = per_env_success
					print(f"  New best candidate! Success rate: {success_rate:.4f}, Mean reward: {mean_reward:.4f}")
			
			total_samples += args.n_candidates
			
			# Step 4: Update pivot if we found a better candidate
			if best_candidate is not None:
				# Store the winning direction for the next iteration
				winning_direction = (best_candidate - pivot) / np.linalg.norm(best_candidate - pivot) * args.lambda_radius
				pivot = best_candidate
				pivot_success_rate = best_candidate_success_rate
				print(f"Updated pivot. New success rate: {pivot_success_rate:.4f}")
			else:
				# No improvement, clear winning direction
				winning_direction = None
				print(f"No improvement found. Keeping current pivot with success rate: {pivot_success_rate:.4f}")
			
			pivot_history.append(pivot.copy())
			pivot_success_history.append(pivot_success_rate)
			
			wandb.log({
				"seed": current_seed,
				"iteration": iteration + 1,
				"pivot_success_rate_updated": pivot_success_rate,
				"pivot_improved": best_candidate is not None,
				"total_samples": total_samples,
				"total_evaluations": total_samples * args.n_envs,
			})
			
			# Save checkpoint when total evaluations reach 1k, 5k, or 10k
			total_evals = total_samples * args.n_envs
			for threshold, label in [(1000, '1k'), (5000, '5k'), (10000, '10k')]:
				flag = f'_saved_{label}_seed_{current_seed}'
				if total_evals >= threshold and not hasattr(main, flag):
					checkpoint_dir = os.path.join(seed_out, f"checkpoint_{label}_evals_iter{iteration+1}_envs{args.n_envs}_cand{args.n_candidates}")
					save_results(checkpoint_dir, all_noise, all_rewards, all_success, 
							    all_success_rates, all_lengths, current_seed, fixed_seeds, noise_idx=None)
					np.save(os.path.join(checkpoint_dir, "pivot_history.npy"), np.array(pivot_history))
					np.save(os.path.join(checkpoint_dir, "pivot_success_history.npy"), np.array(pivot_success_history))
					print(f"Saved checkpoint at {total_evals} evaluations ({label}) at iteration {iteration+1}")
					setattr(main, flag, True)		# Final save
		print(f"\n=== Search Complete ===")
		success_rates_sorted = save_results(seed_out, all_noise, all_rewards, all_success, 
										   all_success_rates, all_lengths, current_seed, fixed_seeds, noise_idx=None)
		
		# Save pivot history
		np.save(os.path.join(seed_out, "pivot_history.npy"), np.array(pivot_history))
		np.save(os.path.join(seed_out, "pivot_success_history.npy"), np.array(pivot_success_history))
		
		# Save ZO-specific summary
		zo_summary = {
			"n_iterations": args.n_iterations,
			"n_candidates_per_iter": args.n_candidates,
			"lambda_radius": args.lambda_radius,
			"distance_metric": args.distance_metric,
			"total_evaluations": total_samples,
			"final_pivot_success_rate": float(pivot_success_rate),
			"pivot_success_history": [float(x) for x in pivot_success_history],
			"best_overall_success_rate": float(success_rates_sorted[0]),
		}
		
		with open(os.path.join(seed_out, "zo_summary.json"), "w") as f:
			json.dump(zo_summary, f, indent=2)
		
		env.close()
		
		print(f"Results saved to {seed_out}")
		print(f"Final pivot success rate: {pivot_success_rate:.4f}")
		print(f"Best overall success rate: {success_rates_sorted[0]:.4f}")
		print(f"Total evaluations: {total_samples}")
		
		wandb.log({
			"seed": current_seed,
			"final_pivot_success_rate": pivot_success_rate,
			"best_overall_success_rate": success_rates_sorted[0],
			"total_samples": total_samples,
			"total_evaluations": total_samples * args.n_envs,
		})
	
	print(f"\n{'='*80}")
	print(f"All {args.n_seeds} seed(s) completed!")
	print(f"{'='*80}")
	
	# Finish wandb after all seeds
	wandb.finish()

if __name__ == "__main__":
	main()

