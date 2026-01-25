import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

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
	p.add_argument("--task_name", default="lift", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100, help="Number of evaluations per seed")
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/noise_eval_results/")
	p.add_argument("--eval", type=str, default=None)
	p.add_argument("--eval_idx", type=int, default=0)
	p.add_argument("--perturbation_magnitudes", type=float, nargs='*', default=[0.0, 0.05, 0.1, 0.25, 0.5], help="Magnitudes of perturbations to test")
	p.add_argument("--n_perturbations", type=int, default=10, help="Number of perturbation samples per magnitude")
	p.add_argument("--n_gradients", type=int, default=10, help="Number of gradient directions to sample")
	p.add_argument("--gradient_magnitudes", type=float, nargs='*', default=[0.0, 0.25, 0.5, 0.75, 1], help="Magnitudes to test along gradient directions")
	return p.parse_args()

def generate_perturbations(base_noise, magnitude, n_perturbations, seed=None):
	"""Generate multiple perturbed noises by adding Gaussian noise."""
	if seed is not None:
		rng = np.random.RandomState(seed)
	else:
		rng = np.random
	
	perturbed_noises = []
	for _ in range(n_perturbations):
		perturbation = rng.normal(0, magnitude, size=base_noise.shape)
		perturbed_noise = base_noise + perturbation
		perturbed_noises.append(perturbed_noise)
	
	return perturbed_noises

def generate_gradient_perturbations(base_noise, gradient_directions, magnitudes):
	"""Generate perturbed noises along specified gradient directions and magnitudes."""
	perturbed_noises = {}
	for grad_idx, grad_dir in enumerate(gradient_directions):
		for mag in magnitudes:
			perturbed_noise = base_noise + mag * grad_dir
			perturbed_noises[(grad_idx, mag)] = perturbed_noise
	return perturbed_noises

def save_eval(out_dir, per_env_rewards_all, episode_success_flags, episode_seeds):
	os.makedirs(out_dir, exist_ok=True)
	# Convert to arrays
	reward_matrix = np.array(per_env_rewards_all, dtype=np.float32)  # shape (episodes, n_envs)
	success_matrix = np.array(episode_success_flags, dtype=bool)     # shape (episodes, n_envs)

	# Per-episode means
	episode_reward_means = reward_matrix.mean(axis=1) if reward_matrix.size else np.array([])
	episode_success_means = success_matrix.mean(axis=1) if success_matrix.size else np.array([])

	# Overall stats (mean and std of per-episode means)
	reward_mean = float(episode_reward_means.mean()) if episode_reward_means.size else 0.0
	reward_std = float(episode_reward_means.std(ddof=1)) if episode_reward_means.size else 0.0
	success_mean = float(episode_success_means.mean()) if episode_success_means.size else 0.0
	success_std = float(episode_success_means.std(ddof=1)) if episode_success_means.size else 0.0

	# Persist full data
	np.save(os.path.join(out_dir, "reward_matrix.npy"), reward_matrix)
	np.save(os.path.join(out_dir, "success_matrix.npy"), success_matrix)
	np.save(os.path.join(out_dir, "episode_reward_means.npy"), episode_reward_means)
	np.save(os.path.join(out_dir, "episode_success_means.npy"), episode_success_means)

	with open(os.path.join(out_dir, "summary.json"), "w") as f:
		json.dump({
			"num_episodes": int(reward_matrix.shape[0]) if reward_matrix.size else 0,
			"num_envs": int(reward_matrix.shape[1]) if reward_matrix.size else 0,
			"reward_mean": reward_mean,
			"reward_std": reward_std,
			"success_mean": success_mean,
			"success_std": success_std,
			"episode_seeds": episode_seeds,
			"per_episode_reward_means": episode_reward_means.tolist(),
			"per_episode_success_means": episode_success_means.tolist(),
			"reward_matrix": reward_matrix.tolist(),
			"success_matrix": success_matrix.astype(bool).tolist(),
		}, f)

	return reward_mean, reward_std, success_mean, success_std

def save_perturbation_results(out_dir, magnitude_results, perturbation_magnitudes):
	os.makedirs(out_dir, exist_ok=True)
	
	# Aggregate statistics per magnitude
	summary = {}
	for mag in perturbation_magnitudes:
		if mag not in magnitude_results:
			continue
		
		results = magnitude_results[mag]
		reward_means = [r[0] for r in results]
		success_means = [r[2] for r in results]
		
		summary[str(mag)] = {
			"reward_mean": float(np.mean(reward_means)),
			"reward_std": float(np.std(reward_means)),
			"success_mean": float(np.mean(success_means)),
			"success_std": float(np.std(success_means)),
			"n_perturbations": len(results),
			"all_reward_means": reward_means,
			"all_success_means": success_means,
		}
	
	with open(os.path.join(out_dir, "perturbation_summary.json"), "w") as f:
		json.dump(summary, f, indent=2)
	
	return summary

def plot_perturbation_results(out_dir, summary, base_reward_mean, base_success_mean):
	os.makedirs(out_dir, exist_ok=True)
	
	magnitudes = sorted([float(k) for k in summary.keys()])
	reward_means = [summary[str(m)]["reward_mean"] for m in magnitudes]
	reward_stds = [summary[str(m)]["reward_std"] for m in magnitudes]
	success_means = [summary[str(m)]["success_mean"] for m in magnitudes]
	success_stds = [summary[str(m)]["success_std"] for m in magnitudes]
	
	# Plot rewards
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	ax1.bar(range(len(magnitudes)), reward_means, yerr=reward_stds, capsize=5, alpha=0.7, color='steelblue', label='Perturbed')
	ax1.axhline(y=base_reward_mean, color='r', linestyle='--', linewidth=2, label=f'Base Noise (mag=0.0)')
	ax1.set_xticks(range(len(magnitudes)))
	ax1.set_xticklabels([f'{m:.2f}' for m in magnitudes])
	ax1.set_xlabel('Perturbation Magnitude', fontsize=12)
	ax1.set_ylabel('Reward (Mean ± Std)', fontsize=12)
	ax1.set_title('Reward vs Perturbation Magnitude', fontsize=14)
	ax1.legend()
	ax1.grid(True, alpha=0.3, axis='y')
	
	# Plot success rates
	ax2.bar(range(len(magnitudes)), success_means, yerr=success_stds, capsize=5, alpha=0.7, color='green', label='Perturbed')
	ax2.axhline(y=base_success_mean, color='r', linestyle='--', linewidth=2, label=f'Base Noise (mag=0.0)')
	ax2.set_xticks(range(len(magnitudes)))
	ax2.set_xticklabels([f'{m:.2f}' for m in magnitudes])
	ax2.set_xlabel('Perturbation Magnitude', fontsize=12)
	ax2.set_ylabel('Success Rate (Mean ± Std)', fontsize=12)
	ax2.set_title('Success Rate vs Perturbation Magnitude', fontsize=14)
	ax2.legend()
	ax2.grid(True, alpha=0.3, axis='y')
	
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "perturbation_analysis.png"), dpi=150)
	plt.close()
	
	print(f"Saved plot to {os.path.join(out_dir, 'perturbation_analysis.png')}")

def plot_gradient_results(out_dir, gradient_results, gradient_magnitudes, base_reward_mean, base_success_mean, n_gradients):
	"""Plot results for gradient-based perturbations in a grid layout."""
	os.makedirs(out_dir, exist_ok=True)
	
	# Organize data by gradient index
	n_cols = min(5, n_gradients)
	n_rows = (n_gradients + n_cols - 1) // n_cols
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
	if n_gradients == 1:
		axes = np.array([[axes]])
	elif n_rows == 1:
		axes = axes.reshape(1, -1)
	
	magnitudes = sorted(gradient_magnitudes)
	
	for grad_idx in range(n_gradients):
		row, col = grad_idx // n_cols, grad_idx % n_cols
		ax = axes[row, col]
		
		# Collect data for this gradient
		rewards = []
		successes = []
		for mag in magnitudes:
			if (grad_idx, mag) in gradient_results:
				r_mean, r_std, s_mean, s_std = gradient_results[(grad_idx, mag)]
				rewards.append(s_mean)  # Using success as primary metric
			else:
				rewards.append(0.0)
		
		# Plot
		ax.scatter(range(len(magnitudes)), rewards, s=100, marker='*', color='purple', alpha=0.7, zorder=3)
		ax.axhline(y=base_success_mean, color='r', linestyle='--', linewidth=2)
		ax.set_xticks(range(len(magnitudes)))
		ax.set_xticklabels([f'{m:.2f}' for m in magnitudes], fontsize=8)
		ax.set_xlabel('Magnitude', fontsize=9)
		ax.set_ylabel('Success Rate', fontsize=9)
		ax.set_title(f'Gradient {grad_idx + 1}', fontsize=10)
		ax.grid(True, alpha=0.3, axis='y')
	
	# Hide unused subplots
	for idx in range(n_gradients, n_rows * n_cols):
		row, col = idx // n_cols, idx % n_cols
		axes[row, col].axis('off')
	
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "gradient_analysis.png"), dpi=150)
	plt.close()
	
	print(f"Saved gradient plot to {os.path.join(out_dir, 'gradient_analysis.png')}")

def evaluate_perturbed_noise_serial(perturbed_noise, base_policy, cfg, video_dir, rew_offset,
                                    n_seeds, n_evals_per_seed, start_seed, save_vid, noise_idx):
	"""Evaluate a perturbed noise using serial rollouts."""

	rewards_all = []
	success_flags_all = []
	episode_seeds = []
	
	for seed_idx in range(n_seeds):
		current_seed = start_seed + seed_idx
		episode_seeds.append(current_seed)
		
		# Build new environment for this seed
		env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
		noise_vec_clipped = np.clip(perturbed_noise.astype(np.float32).flatten(), 
		                             env.action_space.low, env.action_space.high)
		
		# Collect results for this seed
		seed_rewards = []
		seed_successes = []
		
		for eval_idx in range(n_evals_per_seed):
			episode_reward, success = evaluate_noise_single(
				env, noise_vec_clipped, save_vid, noise_idx=noise_idx, 
				eval_num=eval_idx, rew_offset=rew_offset
			)
			seed_rewards.append(float(episode_reward))
			seed_successes.append(bool(success))
		
		rewards_all.append(seed_rewards)
		success_flags_all.append(seed_successes)
		env.close()
	
	return rewards_all, success_flags_all, episode_seeds

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
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	best_noise = load_noise_idx(args.eval, args.eval_idx)
	print(f">>> Loaded noise idx {args.eval_idx} from {args.eval}")
	
	# Extract ticket name from eval path
	ticket_name = os.path.basename(args.eval.rstrip('/')) if args.eval else f"noise_{args.eval_idx}"
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"{ticket_name}_perturb_idx{args.eval_idx}_nmag{len(args.perturbation_magnitudes)}_npert{args.n_perturbations}_ddim{cfg.model.ddim_steps}_seed{args.seed}_{timestamp}"
	args.out = os.path.join(args.out, args.task_name, run_name)
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = False  # Disable video saving for perturbation analysis
	
	# Dictionary to store results for each magnitude
	magnitude_results = {}
	base_reward_mean = None
	base_success_mean = None
	
	# Loop over perturbation magnitudes
	for mag_idx, magnitude in enumerate(args.perturbation_magnitudes):
		print(f"\n{'='*60}")
		print(f"Evaluating perturbation magnitude: {magnitude}")
		print(f"{'='*60}")
		
		# Generate perturbations for this magnitude
		if magnitude == 0.0:
			# Base noise (no perturbation)
			perturbed_noises = [best_noise]
			n_perturb_for_mag = 1
		else:
			perturbed_noises = generate_perturbations(best_noise, magnitude, args.n_perturbations, seed=cfg.seed + mag_idx)
			n_perturb_for_mag = args.n_perturbations
		
		magnitude_results[magnitude] = []
		
		# Evaluate each perturbed noise using serial rollouts
		for pert_idx, perturbed_noise in enumerate(perturbed_noises):
			print(f"\n--- Magnitude {magnitude}, Perturbation {pert_idx + 1}/{n_perturb_for_mag} ---")
			
			# Serial rollouts: n_seeds x n_evals_per_seed
			per_seed_rewards_all, per_seed_success_flags, episode_seeds = evaluate_perturbed_noise_serial(
				perturbed_noise, base_policy, cfg, video_dir, cfg.env.reward_offset,
				args.n_seeds, args.n_evals_per_seed, args.seed, save_vid, args.eval_idx
			)
			
			# Save results for this perturbation
			pert_out_dir = os.path.join(args.out, f"magnitude_{magnitude}", f"perturbation_{pert_idx}")
			reward_mean, reward_std, success_mean, success_std = save_eval_serial(
				pert_out_dir, per_seed_rewards_all, per_seed_success_flags, episode_seeds
			)
			
			magnitude_results[magnitude].append((reward_mean, reward_std, success_mean, success_std))
			
			print(f"  Reward: {reward_mean:.4f} ± {reward_std:.4f}, Success: {success_mean:.4f} ± {success_std:.4f}")
			
			# Store base noise performance for plotting
			if magnitude == 0.0:
				base_reward_mean = reward_mean
				base_success_mean = success_mean
	
	# Save aggregated results and create plots
	summary = save_perturbation_results(args.out, magnitude_results, args.perturbation_magnitudes)
	
	if base_reward_mean is not None and base_success_mean is not None:
		plot_perturbation_results(args.out, summary, base_reward_mean, base_success_mean)
	
	# Gradient-based perturbations experiment
	print(f"\n{'='*60}")
	print("Starting gradient-based perturbation experiment")
	print(f"{'='*60}")
	
	# Sample random gradient directions
	rng = np.random.RandomState(cfg.seed + 1000)
	gradient_directions = [rng.randn(*best_noise.shape) for _ in range(args.n_gradients)]
	gradient_directions = [g / (np.linalg.norm(g) + 1e-8) for g in gradient_directions]  # Normalize
	
	# Generate all gradient perturbations
	gradient_perturbed_noises = generate_gradient_perturbations(best_noise, gradient_directions, args.gradient_magnitudes)
	
	gradient_results = {}
	
	for (grad_idx, magnitude), perturbed_noise in gradient_perturbed_noises.items():
		print(f"\n--- Gradient {grad_idx + 1}/{args.n_gradients}, Magnitude {magnitude} ---")
		
		# Serial rollouts for gradient perturbations
		per_seed_rewards_all, per_seed_success_flags, episode_seeds = evaluate_perturbed_noise_serial(
			perturbed_noise, base_policy, cfg, video_dir, cfg.env.reward_offset,
			args.n_seeds, args.n_evals_per_seed, args.seed, False, args.eval_idx
		)
		
		grad_out_dir = os.path.join(args.out, f"gradient_{grad_idx}", f"magnitude_{magnitude}")
		reward_mean, reward_std, success_mean, success_std = save_eval_serial(
			grad_out_dir, per_seed_rewards_all, per_seed_success_flags, episode_seeds
		)
		
		gradient_results[(grad_idx, magnitude)] = (reward_mean, reward_std, success_mean, success_std)
		print(f"  Reward: {reward_mean:.4f} ± {reward_std:.4f}, Success: {success_mean:.4f} ± {success_std:.4f}")
	
	# Save gradient results and plot
	with open(os.path.join(args.out, "gradient_results.json"), "w") as f:
		json.dump({f"{k[0]}_{k[1]}": v for k, v in gradient_results.items()}, f, indent=2)
	
	if base_reward_mean is not None and base_success_mean is not None:
		plot_gradient_results(args.out, gradient_results, args.gradient_magnitudes, base_reward_mean, base_success_mean, args.n_gradients)
	
	print(f"\n{'='*60}")
	print("Evaluation complete!")
	print(f"Results saved to: {args.out}")
	print(f"{'='*60}")

if __name__ == "__main__":
	main()
