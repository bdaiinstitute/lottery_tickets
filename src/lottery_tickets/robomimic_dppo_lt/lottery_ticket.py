"""
- Noise search is configured in a way that every reset of the environment resets to the same seed
assigned to that environment instance. All the environment instances are stepped in parallel,
while the noise instances are evaluated in sequence over all the environment instances.
- Unlike DMG, we don't zero the Gaussian noise added to manipulator initial state
- Reward calculation is different from the way DSRL does it for RM envs as ignore once finished
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

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from env_util import build_lt_env
from policy_util import load_base_policy

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="lift", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_envs", type=int, default=100)
	p.add_argument("--noise_samples", type=int, default=5000)
	p.add_argument("--seed", type=int, default=999)
	p.add_argument("--exp_name", type=str, default="", help="Experiment name to append to output folder")
	p.add_argument("--out", default="logs_res_rm/lottery_ticket_results/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_envs: int, noise_samples: int, seed: int, ddim_steps: int, exp_name: str = "") -> str:
	ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
	run_name = f"envs{n_envs}_samples{noise_samples}_seed{seed}_ddim{ddim_steps}_{ts}"
	if exp_name:
		run_name = f"{run_name}_{exp_name}"
	return os.path.join(out_path.rstrip('/'), task_name, run_name)


def save_results(out_dir, all_noise, all_rewards, all_success, all_success_rates, all_lengths, main_seed=None, env_seeds=None, noise_idx=None):
	"""Persist full noise search results with ranking and summary statistics."""
	# Create subdirectory with noise index if provided
	if noise_idx is not None:
		out_dir = os.path.join(out_dir, f"checkpoint_{noise_idx}")
	os.makedirs(out_dir, exist_ok=True)
	success_rates_arr = np.array(all_success_rates, dtype=np.float32)
	
	# Calculate mean reward for each noise sample
	mean_rewards = np.array([np.mean(rewards) for rewards in all_rewards], dtype=np.float32)
	
	# Sort by success rate (descending), then by mean reward (descending)
	# lexsort reads keys from bottom to top
	order = np.lexsort((-mean_rewards, -success_rates_arr))

	noise_sorted = np.array(all_noise, dtype=np.float32)[order]
	rewards_sorted = [all_rewards[i] for i in order]
	success_sorted = [all_success[i] for i in order]
	success_rates_sorted = [all_success_rates[i] for i in order]
	lengths_sorted = [all_lengths[i] for i in order]

	# Save raw sorted arrays
	np.save(os.path.join(out_dir, "noise_samples.npy"), noise_sorted)
	np.save(os.path.join(out_dir, "rewards.npy"), np.array(rewards_sorted, dtype=object))
	np.save(os.path.join(out_dir, "successes.npy"), np.array(success_sorted, dtype=object))
	np.save(os.path.join(out_dir, "success_rates.npy"), np.array(success_rates_sorted, dtype=np.float32))
	np.save(os.path.join(out_dir, "episode_lengths.npy"), np.array(lengths_sorted, dtype=object))

	# JSON serialization (convert numpy types to native Python)
	success_serializable = [[bool(x) for x in row] for row in success_sorted]

	results_dict = {
		"noise": noise_sorted.tolist(),
		"rewards": rewards_sorted,
		"successes": success_serializable,
		"success_rates": success_rates_sorted,
		"episode_lengths": lengths_sorted,
	}

	with open(os.path.join(out_dir, "results.json"), "w") as f:
		json.dump(results_dict, f)

	with open(os.path.join(out_dir, "ranking.txt"), "w") as f:
		for rank, i in enumerate(order):
			mean_rew = np.mean(all_rewards[i])
			f.write(f"{rank}\tidx={i}\tsuccess_rate={all_success_rates[i]:.4f}\tmean_reward={mean_rew:.4f}\n")

	# Aggregated summary similar to evaluation script parity
	summary = {
		"num_noise_samples": int(len(all_noise)),
		"num_envs": int(len(all_rewards[0])) if all_rewards else 0,
		"success_rate_mean": float(success_rates_arr.mean()) if success_rates_arr.size else 0.0,
		"success_rate_std": float(success_rates_arr.std(ddof=1)) if success_rates_arr.size else 0.0,
		"best_success_rate": float(success_rates_sorted[0]) if success_rates_sorted else 0.0,
		"best_original_index": int(order[0]) if success_rates_sorted else -1,
		"success_rates_sorted": success_rates_sorted,
		"original_success_rates": all_success_rates,
	}
	if main_seed is not None:
		summary["seed"] = main_seed
	if env_seeds is not None:
		summary["env_seeds"] = env_seeds
	
	with open(os.path.join(out_dir, "summary.json"), "w") as f:
		json.dump(summary, f)

	return success_rates_sorted

def evaluate_noise(env, noise_vec, n_envs, save_vid=False, noise_idx=0, rew_offset=0.0, expected_initial_obs=None):
	"""Evaluate a single noise vector across parallel envs.

	Mirrors the reward & success accounting used in utils.LoggingCallback.evaluate:
	- Accumulates episode reward until done per env
	- Success flagged if any step reward > -rew_offset
	- Stops when all envs finished or max_steps reached
	Returns per-env episode rewards and success booleans.
	"""
	if save_vid:
		env.env.name_prefix = f"noise_{noise_idx}"

	obs = env.reset()
	
	# Verify initial observation matches expected if provided
	if expected_initial_obs is not None:
		assert np.allclose(obs, expected_initial_obs, rtol=1e-5, atol=1e-6), \
			f"Initial observation mismatch at noise_idx={noise_idx}. Expected consistent initial states."

	# Storage similar to evaluate() logic
	per_env_reward = [0.0] * n_envs          # Final episode reward per env
	success_flag = [False] * n_envs          # Success per env (any step)
	per_env_length = [0] * n_envs            # Episode length per env
	finished = [False] * n_envs
	rew_ep = np.zeros(n_envs, dtype=np.float32)  # Running reward until each env is done

	# Same noise actions fed every macro step (wrapper handles diffusion policy)
	# If noise_vec is batched (has batch dim), use it directly; otherwise repeat single vector
	if noise_vec.ndim == 2 and noise_vec.shape[0] == n_envs:
		actions = noise_vec
	else:
		actions = np.repeat(noise_vec.reshape(1, -1), n_envs, axis=0)

	steps = 0
	while not all(finished):
		_, r, d, info = env.step(actions)
		steps += 1

		r_np = np.array(r, dtype=np.float32)
		
		for i in range(n_envs):
			if finished[i]:
				continue
			per_env_length[i] = steps
			rew_ep[i] += r_np[i]
			if r_np[i] > -rew_offset:
				success_flag[i] = True
				assert d[i], "Should have been set upstream in ActionChunkWrapper"
			if d[i]:
				per_env_reward[i] = float(rew_ep[i])
				rew_ep[i] = 0.0
				finished[i] = True

	# For any env that did not finish, record partial accumulated reward
	for i in range(n_envs):
		if not finished[i]:
			per_env_reward[i] = float(rew_ep[i])
			per_env_length[i] = steps

	return per_env_reward, success_flag, per_env_length

def main():
	args = p_args()
	base_path = str(BASE_DIR)
	config_path = os.path.join(f"{base_path}/src/lottery_tickets/robomimic_dppo_lt", TASK_CONFIGS[args.task_name])
	
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
	
	args.out = _resolve_out(args.out, args.task_name, args.n_envs, args.noise_samples, args.seed, cfg.model.ddim_steps, args.exp_name)
	
	# Initialize wandb
	if not args.no_wandb:
		run_name = args.task_name + "_" + os.path.basename(args.out)
		wandb.init(
			project="lottery_ticket_rm",
			name=run_name,
			config={
				"task_name": args.task_name,
				"n_envs": args.n_envs,
				"noise_samples": args.noise_samples,
				"seed": args.seed,
				"ddim_steps": cfg.model.ddim_steps,
				"exp_name": args.exp_name,
				"output_dir": args.out,
			},
			tags=[args.task_name, "lottery_ticket"],
		)
	
	if not hasattr(cfg, 'device') or cfg.device is None:
		cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
	
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = args.n_envs < 10
	# Derive and fix per-env seeds once so every reset keeps the same mapping
	ss_env = np.random.SeedSequence(args.seed)
	fixed_seeds = [int(s.generate_state(1)[0]) for s in ss_env.spawn(args.n_envs)]
	env = build_lt_env(base_policy, cfg, args.n_envs, video_dir, save_vid=save_vid, fixed_seeds=fixed_seeds)
	
	# Create RNG for noise generation
	rng = np.random.default_rng(args.seed)
	
	all_noise = []
	all_rewards = []
	all_success = []
	all_success_rates = []
	all_lengths = []
	
	# Track running statistics for wandb
	best_success_rate = 0.0
	best_mean_reward = 0.0
	best_idx = -1
	
	# Save initial observation for verification
	initial_obs = env.reset()
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	np.save(os.path.join(args.out, "initial_obs.npy"), initial_obs)
	print(f"Saved initial observation with shape {initial_obs.shape}")
	
	# Get action space dimension from the wrapped environment
	action_space_dim = env.action_space.shape[0]  # This is act_steps * action_dim
	
	for noise_idx in range(args.noise_samples):
		print(f"Starting noise sample {noise_idx}")
		noise_vec = rng.standard_normal(action_space_dim).astype(np.float32)

		# ideally should not be clipped, but kept for backcompatibility
		# noise_vec = np.clip(noise_vec, env.action_space.low, env.action_space.high)
		per_env_reward, per_env_success, per_env_length = evaluate_noise(
			env, noise_vec, args.n_envs, save_vid, noise_idx, rew_offset=cfg.env.reward_offset,
			expected_initial_obs=initial_obs
		)
		
		all_noise.append(noise_vec)
		all_rewards.append(per_env_reward)
		all_success.append(per_env_success)
		all_lengths.append(per_env_length)
		current_success_rate = float(np.mean(per_env_success))
		all_success_rates.append(current_success_rate)
		
		current_mean_reward = np.mean(per_env_reward)
		if (current_success_rate > best_success_rate or 
			(current_success_rate == best_success_rate and current_mean_reward > best_mean_reward)):
			best_success_rate = current_success_rate
			best_mean_reward = current_mean_reward
			best_idx = noise_idx
		
		# Log to wandb
		if not args.no_wandb:
			wandb.log({
				"iteration": noise_idx,
				"success_rate": current_success_rate,
				"mean_reward": current_mean_reward,
				"std_reward": np.std(per_env_reward),
				"mean_episode_length": np.mean(per_env_length),
				"std_episode_length": np.std(per_env_length),
				"best_success_rate_so_far": best_success_rate,
				"best_mean_reward_so_far": best_mean_reward,
				"best_idx_so_far": best_idx,
				"num_successes": sum(per_env_success),
			}, step=noise_idx)
		
		if (noise_idx + 1) % 500 == 0:
			save_results(args.out, all_noise, all_rewards, all_success, all_success_rates, all_lengths, args.seed, fixed_seeds, noise_idx=noise_idx)
			print(f"Saved checkpoint at sample {noise_idx + 1}")
	
	success_rates_sorted = save_results(args.out, all_noise, all_rewards, all_success, all_success_rates, all_lengths, args.seed, fixed_seeds, noise_idx=None)
	env.close()
	
	# Log final summary to wandb
	if not args.no_wandb:
		final_summary = {
			"final/best_success_rate": success_rates_sorted[0],
			"final/best_mean_reward": best_mean_reward,
			"final/best_idx": best_idx,
			"final/mean_success_rate": np.mean(all_success_rates),
			"final/std_success_rate": np.std(all_success_rates),
			"final/median_success_rate": np.median(all_success_rates),
			"final/num_samples": len(all_success_rates),
			"final/top_10_mean": np.mean(success_rates_sorted[:10]) if len(success_rates_sorted) >= 10 else np.mean(success_rates_sorted),
		}
		wandb.log(final_summary)
		wandb.log({"success_rate_distribution": wandb.Histogram(all_success_rates)})
		wandb.finish()
	
	print(f"Search complete. Results saved to {args.out}")
	print(f"Best success rate: {success_rates_sorted[0]:.4f}")

if __name__ == "__main__":
	main()
