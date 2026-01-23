"""Stage-2 lottery ticket finder"""

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

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_lt_env, load_base_policy
from st1_ticket import TASK_CONFIGS, STAGE_THRESHOLDS

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="square", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_envs", type=int, default=100)
	p.add_argument("--noise_samples", type=int, default=1000)
	p.add_argument("--seed", type=int, default=999)
	p.add_argument("--st1_tk", type=str, default=None, help="Path to stage 1 tickets directory (required for stage 2)")
	p.add_argument("--out", default="logs_res_rm/lottery_ticket_results/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_envs: int, noise_samples: int, seed: int, ddim_steps: int, reward_thresholds: list = None) -> str:
	ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
	threshold_str = f"thresh{'-'.join(map(str, reward_thresholds.values()))}_" if reward_thresholds is not None else ""
	run_name = f"stage2_{threshold_str}envs{n_envs}_samples{noise_samples}_seed{seed}_ddim{ddim_steps}_{ts}"
	return os.path.join(out_path.rstrip('/'), task_name, run_name)


def save_results(out_dir, all_st1_noise, all_st2_noise, all_rewards, all_success, all_success_rates, all_st1_success, all_st1_success_rates, all_lengths, reward_thresholds, main_seed=None, env_seeds=None):
	"""Persist full noise search results with ranking and summary statistics."""
	os.makedirs(out_dir, exist_ok=True)
	success_rates_arr = np.array(all_success_rates, dtype=np.float32)
	mean_rewards = np.array([np.mean(rewards) for rewards in all_rewards], dtype=np.float32)
	order = np.lexsort((-mean_rewards, -success_rates_arr))
	
	st1_noise_sorted = np.array(all_st1_noise, dtype=np.float32)[order]
	st2_noise_sorted = np.array(all_st2_noise, dtype=np.float32)[order]
	rewards_sorted = [all_rewards[i] for i in order]
	success_sorted = [all_success[i] for i in order]
	success_rates_sorted = [all_success_rates[i] for i in order]
	st1_success_sorted = [all_st1_success[i] for i in order]
	st1_success_rates_sorted = [all_st1_success_rates[i] for i in order]
	lengths_sorted = [all_lengths[i] for i in order]

	all_st2_indices = list(range(len(all_st2_noise)))
	st2_indices_sorted = [all_st2_indices[i] for i in order]

	# Save raw sorted arrays
	np.save(os.path.join(out_dir, "st1_noise_samples.npy"), st1_noise_sorted)
	np.save(os.path.join(out_dir, "st2_noise_samples.npy"), st2_noise_sorted)
	np.save(os.path.join(out_dir, "st2_indices.npy"), np.array(st2_indices_sorted, dtype=np.int32))
	np.save(os.path.join(out_dir, "rewards.npy"), np.array(rewards_sorted, dtype=object))
	np.save(os.path.join(out_dir, "successes.npy"), np.array(success_sorted, dtype=object))
	np.save(os.path.join(out_dir, "success_rates.npy"), np.array(success_rates_sorted, dtype=np.float32))
	np.save(os.path.join(out_dir, "st1_successes.npy"), np.array(st1_success_sorted, dtype=object))
	np.save(os.path.join(out_dir, "st1_success_rates.npy"), np.array(st1_success_rates_sorted, dtype=np.float32))
	np.save(os.path.join(out_dir, "episode_lengths.npy"), np.array(lengths_sorted, dtype=object))

	# JSON serialization (convert numpy types to native Python)
	success_serializable = [[bool(x) for x in row] for row in success_sorted]
	st1_success_serializable = [[bool(x) for x in row] for row in st1_success_sorted]

	with open(os.path.join(out_dir, "results.json"), "w") as f:
		json.dump(
			{
				"st1_noise": st1_noise_sorted.tolist(),
				"st2_noise": st2_noise_sorted.tolist(),
				"st2_indices": st2_indices_sorted,
				"rewards": rewards_sorted,
				"successes": success_serializable,
				"success_rates": success_rates_sorted,
				"st1_successes": st1_success_serializable,
				"st1_success_rates": st1_success_rates_sorted,
				"episode_lengths": lengths_sorted,
				"reward_thresholds": reward_thresholds,
			},
			f,
		)

	with open(os.path.join(out_dir, "ranking.txt"), "w") as f:
		for rank, i in enumerate(order):
			mean_reward = np.mean(all_rewards[i])
			f.write(f"{rank}\tidx={i}\tst2_idx={all_st2_indices[i]}\tmean_reward={mean_reward:.4f}\tsuccess_rate={all_success_rates[i]:.4f}\tst1_success_rate={all_st1_success_rates[i]:.4f}\n")

	# Aggregated summary similar to evaluation script parity
	success_rates_arr = np.array(all_success_rates, dtype=np.float32)
	st1_success_rates_arr = np.array(all_st1_success_rates, dtype=np.float32)
	summary = {
		"num_noise_samples": int(len(all_st2_noise)),
		"num_envs": int(len(all_rewards[0])) if all_rewards else 0,
		"success_rate_mean": float(success_rates_arr.mean()) if success_rates_arr.size else 0.0,
		"success_rate_std": float(success_rates_arr.std(ddof=1)) if success_rates_arr.size else 0.0,
		"best_success_rate": float(success_rates_sorted[0]) if success_rates_sorted else 0.0,
		"best_original_index": int(order[0]) if success_rates_sorted else -1,
		"best_st2_index": int(st2_indices_sorted[0]) if st2_indices_sorted else -1,
		"success_rates_sorted": success_rates_sorted,
		"original_success_rates": all_success_rates,
		"st1_success_rate_mean": float(st1_success_rates_arr.mean()) if st1_success_rates_arr.size else 0.0,
		"st1_success_rate_std": float(st1_success_rates_arr.std(ddof=1)) if st1_success_rates_arr.size else 0.0,
		"st1_success_rates_sorted": st1_success_rates_sorted,
		"original_st1_success_rates": all_st1_success_rates,
		"reward_thresholds": reward_thresholds,
	}
	if main_seed is not None:
		summary["seed"] = main_seed
	if env_seeds is not None:
		summary["env_seeds"] = env_seeds
	
	with open(os.path.join(out_dir, "summary.json"), "w") as f:
		json.dump(summary, f)

	return success_rates_sorted

def evaluate_noise(env, st1_noise_vec, st2_noise_vec, n_envs, save_vid=False, noise_idx=0, reward_thresholds=None):
	"""Evaluate a single noise vector across parallel envs.

	Success flagged if cumulative episode reward exceeds reward_threshold.
	Returns per-env episode rewards, success booleans, and episode lengths.
	"""
	if save_vid:
		env.env.name_prefix = f"noise_{noise_idx}"

	env.reset()
	# Storage similar to evaluate() logic
	per_env_reward = [0.0] * n_envs          # Final episode reward per env
	success_flag = [False] * n_envs          # Success per env (any step)
	st1_success = [False] * n_envs
	per_env_length = [0] * n_envs            # Episode length per env
	finished = [False] * n_envs
	rew_ep = np.zeros(n_envs, dtype=np.float32)  # Running reward until each env is done

	actions = np.repeat(st1_noise_vec.reshape(1, -1), n_envs, axis=0) # initialize with st1 noise
	st2_actions = st2_noise_vec.reshape(1, -1)

	steps = 0
	while not all(finished):
		_, r, d, info = env.step(actions)
		steps += env.action_horizon
		r_np = np.array(r, dtype=np.float32)
		
		for i in range(n_envs):
			if finished[i]:
				continue
			per_env_length[i] = steps
			rew_ep[i] += r_np[i]

			if r_np[i] > reward_thresholds[1]:
				actions[i] = st2_actions
				st1_success[i] = True

			if r_np[i] > reward_thresholds[2]:
				success_flag[i] = True
				finished[i] = True # prematurely finish
				per_env_reward[i] = float(rew_ep[i])
				rew_ep[i] = 0.0

			if steps >= env.max_episode_steps and not success_flag[i]: # failed
				per_env_reward[i] = float(rew_ep[i])
				rew_ep[i] = 0.0
				finished[i] = True

	# For any env that did not finish, record partial accumulated reward
	for i in range(n_envs):
		if not finished[i]:
			per_env_reward[i] = float(rew_ep[i])

	return per_env_reward, success_flag, st1_success, per_env_length

def load_previous_tickets(prev_stage_dir):
	"""Load tickets from previous stage sorted by success rate."""
	noise_path = os.path.join(prev_stage_dir, "noise_samples.npy")
	if not os.path.exists(noise_path):
		raise FileNotFoundError(f"Previous stage tickets not found at {noise_path}")
	return np.load(noise_path)

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

	# Allow mutation of config (parity with eval script conveniences)
	OmegaConf.set_struct(cfg, False)
	cfg.seed = args.seed
	
	# Override ddim_steps if provided
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps
	
	reward_thresholds = STAGE_THRESHOLDS[args.task_name]
	args.out = _resolve_out(args.out, args.task_name, args.n_envs, args.noise_samples, args.seed, cfg.model.ddim_steps, reward_thresholds)
	
	# Initialize wandb
	if not args.no_wandb:
		run_name = f"{args.task_name}_st2_{os.path.basename(args.out)}"
		wandb.init(
			project="lottery_ticket_rm",
			name=run_name,
			config={
				"task_name": args.task_name,
				"stage": 2,
				"n_envs": args.n_envs,
				"noise_samples": args.noise_samples,
				"seed": args.seed,
				"ddim_steps": cfg.model.ddim_steps,
				"reward_thresholds": reward_thresholds,
				"st1_tk_path": args.st1_tk,
				"output_dir": args.out,
			},
			tags=[args.task_name, "lottery_ticket", "stage2"],
		)
	
	if not hasattr(cfg, 'device') or cfg.device is None:
		cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
	
	# Load previous stage tickets
	if args.st1_tk is None:
		raise ValueError("--st1_tk path is required for stage 2")
	if not os.path.exists(args.st1_tk):
		raise FileNotFoundError(f"Stage 1 tickets path does not exist: {args.st1_tk}")
	prev_tickets = load_previous_tickets(args.st1_tk)
	print(f"Loaded {len(prev_tickets)} tickets from stage 1: {args.st1_tk}")
	
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = args.n_envs < 10
	# Derive and fix per-env seeds once so every reset keeps the same mapping
	ss_env = np.random.SeedSequence(args.seed)
	fixed_seeds = [int(s.generate_state(1)[0]) for s in ss_env.spawn(args.n_envs)]
	env = build_lt_env(base_policy, cfg, args.n_envs, video_dir, save_vid=save_vid, fixed_seeds=fixed_seeds, reward_shaping=True)
		
	# Save initial observation for verification
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	action_space_dim = env.action_space.shape[0]  # This is act_steps * action_dim
	
	for st1_tk_idx in range(5):
		# Create RNG for noise generation
		rng = np.random.default_rng(args.seed+1)

		# Create subdirectory for this st1 ticket
		st1_subdir = os.path.join(args.out, f"st1_ticket_{st1_tk_idx}")
		st1_noise_vec = prev_tickets[st1_tk_idx]
		os.makedirs(st1_subdir, exist_ok=True)
		
		# Reset storage for this st1 ticket
		all_st2_noise = []
		all_rewards = []
		all_st1_success = []
		all_st1_success_rates = []
		all_success = []
		all_success_rates = []
		all_lengths = []
		
		# Track running statistics for wandb
		best_success_rate = 0.0
		best_mean_reward = 0.0
		best_idx = -1
		
		for noise_idx in range(args.noise_samples):
			print(f"Starting st1_ticket {st1_tk_idx}, noise sample {noise_idx}")
			st2_noise_vec = rng.standard_normal(action_space_dim).astype(np.float32)
			# ideally should not be clipped, but kept for backcompatibility
			# st2_noise_vec = np.clip(st2_noise_vec, env.action_space.low, env.action_space.high)
			per_env_reward, per_env_success, st1_success, per_env_length = evaluate_noise(
				env, st1_noise_vec, st2_noise_vec, args.n_envs, save_vid, noise_idx, reward_thresholds=reward_thresholds
			)
			
			all_st2_noise.append(st2_noise_vec)
			all_rewards.append(per_env_reward)
			all_success.append(per_env_success)
			current_success_rate = float(np.mean(per_env_success))
			all_success_rates.append(current_success_rate)
			# st1 metrics
			all_st1_success.append(st1_success)
			all_st1_success_rates.append(float(np.mean(st1_success)))
			all_lengths.append(per_env_length)
			
			# Update statistics with tie-breaking by reward
			current_mean_reward = np.mean(per_env_reward)
			if (current_success_rate > best_success_rate or 
				(current_success_rate == best_success_rate and current_mean_reward > best_mean_reward)):
				best_success_rate = current_success_rate
				best_mean_reward = current_mean_reward
				best_idx = noise_idx
			
			# Log to wandb (with st1_ticket_idx prefix)
			if not args.no_wandb:
				wandb.log({
					f"st1_{st1_tk_idx}/iteration": noise_idx,
					f"st1_{st1_tk_idx}/success_rate": current_success_rate,
					f"st1_{st1_tk_idx}/mean_reward": current_mean_reward,
					f"st1_{st1_tk_idx}/std_reward": np.std(per_env_reward),
					f"st1_{st1_tk_idx}/mean_episode_length": np.mean(per_env_length),
					f"st1_{st1_tk_idx}/std_episode_length": np.std(per_env_length),
					f"st1_{st1_tk_idx}/best_success_rate_so_far": best_success_rate,
					f"st1_{st1_tk_idx}/best_mean_reward_so_far": best_mean_reward,
					f"st1_{st1_tk_idx}/best_idx_so_far": best_idx,
					f"st1_{st1_tk_idx}/st1_success_rate": float(np.mean(st1_success)),
				}, step=st1_tk_idx * args.noise_samples + noise_idx)
			
			if (noise_idx + 1) % 500 == 0:
				all_st1_noise = [st1_noise_vec] * len(all_st2_noise)
				save_results(st1_subdir, all_st1_noise, all_st2_noise, all_rewards, all_success, all_success_rates, all_st1_success, all_st1_success_rates, all_lengths, reward_thresholds, args.seed, fixed_seeds)
				print(f"Saved checkpoint at st1_ticket {st1_tk_idx}, sample {noise_idx + 1}")
		
		# Save final results for this st1 ticket
		all_st1_noise = [st1_noise_vec] * len(all_st2_noise)
		success_rates_sorted = save_results(st1_subdir, all_st1_noise, all_st2_noise, all_rewards, all_success, all_success_rates, all_st1_success, all_st1_success_rates, all_lengths, reward_thresholds, args.seed, fixed_seeds)
		print(f"Completed st1_ticket {st1_tk_idx}. Best success rate: {success_rates_sorted[0]:.4f}")
		
		# Log per-st1-ticket summary
		if not args.no_wandb:
			st1_summary = {
				f"st1_{st1_tk_idx}/final_best_success_rate": success_rates_sorted[0],
				f"st1_{st1_tk_idx}/final_best_mean_reward": best_mean_reward,
				f"st1_{st1_tk_idx}/final_best_idx": best_idx,
				f"st1_{st1_tk_idx}/final_mean_success_rate": np.mean(all_success_rates),
				f"st1_{st1_tk_idx}/final_std_success_rate": np.std(all_success_rates),
			}
			wandb.log(st1_summary)
	
	env.close()
	
	if not args.no_wandb:
		wandb.finish()
	
	print(f"Search complete. Results saved to {args.out}")

if __name__ == "__main__":
	main()
