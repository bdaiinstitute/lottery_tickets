"""Budget Stage-1 lottery ticket finder with multiple seeds"""

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

BASE_DIR = Path(__file__).resolve().parents[4]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_lt_env, load_base_policy
from noise_opt_rm.ticket_search.rollout.st1_ticket import TASK_CONFIGS, STAGE_THRESHOLDS, evaluate_st1_noise

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="square", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_envs", type=int, default=100)
	p.add_argument("--noise_samples", type=int, default=1000)
	p.add_argument("--seed", type=int, default=999, help="Starting seed")
	p.add_argument("--n_seeds", type=int, default=10, help="Number of seeds")
	p.add_argument("--exp_name", type=str, default="", help="Experiment name to append to output folder")
	p.add_argument("--out", default="logs_res_rm/lottery_ticket_results/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_envs: int, noise_samples: int, seeds: list, ddim_steps: int, reward_threshold: float = None, exp_name: str = "") -> str:
	ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
	seed_str = "_".join(map(str, seeds))
	threshold_str = f"thresh{reward_threshold}_" if reward_threshold is not None else ""
	run_name = f"stage1_{threshold_str}envs{n_envs}_samples{noise_samples}_seeds{seed_str}_ddim{ddim_steps}_{ts}"
	if exp_name:
		run_name = f"{run_name}_{exp_name}"
	return os.path.join(out_path.rstrip('/'), task_name, run_name)

def save_results(out_dir, all_noise, all_rewards, all_success, all_success_rates, all_lengths, main_seed=None, env_seeds=None, noise_idx=None, seed_id=None):
	"""Persist full noise search results with ranking and summary statistics."""
	out_dir = os.path.join(out_dir, f"seed_{seed_id}")
	if noise_idx is not None:
		out_dir = os.path.join(out_dir, f"checkpoint_{noise_idx}")
	os.makedirs(out_dir, exist_ok=True)
	success_rates_arr = np.array(all_success_rates, dtype=np.float32)
	mean_rewards = np.array([np.mean(rewards) for rewards in all_rewards], dtype=np.float32)
	order = np.lexsort((-mean_rewards, -success_rates_arr))

	noise_sorted = np.array(all_noise, dtype=np.float32)[order]
	rewards_sorted = [all_rewards[i] for i in order]
	success_sorted = [all_success[i] for i in order]
	success_rates_sorted = [all_success_rates[i] for i in order]
	lengths_sorted = [all_lengths[i] for i in order]

	np.save(os.path.join(out_dir, "noise_samples.npy"), noise_sorted)
	np.save(os.path.join(out_dir, "rewards.npy"), np.array(rewards_sorted, dtype=object))
	np.save(os.path.join(out_dir, "successes.npy"), np.array(success_sorted, dtype=object))
	np.save(os.path.join(out_dir, "success_rates.npy"), np.array(success_rates_sorted, dtype=np.float32))
	np.save(os.path.join(out_dir, "episode_lengths.npy"), np.array(lengths_sorted, dtype=object))

	success_serializable = [[bool(x) for x in row] for row in success_sorted]
	with open(os.path.join(out_dir, "results.json"), "w") as f:
		json.dump({
			"noise": noise_sorted.tolist(),
			"rewards": rewards_sorted,
			"successes": success_serializable,
			"success_rates": success_rates_sorted,
			"episode_lengths": lengths_sorted,
		}, f)

	with open(os.path.join(out_dir, "ranking.txt"), "w") as f:
		for rank, i in enumerate(order):
			mean_reward = np.mean(all_rewards[i])
			f.write(f"{rank}\tidx={i}\tmean_reward={mean_reward:.4f}\tsuccess_rate={all_success_rates[i]:.4f}\n")

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

def main():
	args = p_args()
	seeds = [args.seed + i for i in range(args.n_seeds)]
	base_path = str(BASE_DIR)
	config_path = os.path.join(base_path, TASK_CONFIGS[args.task_name])
	
	OmegaConf.register_new_resolver("eval", eval)
	config_dir = os.path.dirname(config_path)
	config_name = os.path.basename(config_path).replace('.yaml', '')
	
	with initialize_config_dir(version_base=None, config_dir=config_dir):
		cfg = compose(config_name=config_name)

	OmegaConf.set_struct(cfg, False)
	
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps
	
	reward_threshold = STAGE_THRESHOLDS[args.task_name][1]
	args.out = _resolve_out(args.out, args.task_name, args.n_envs, args.noise_samples, seeds, cfg.model.ddim_steps, reward_threshold, args.exp_name)
	
	if not args.no_wandb:
		run_name = f"{args.task_name}_st1_{os.path.basename(args.out)}"
		wandb.init(
			project="lottery_ticket_rm",
			name=run_name,
			config={
				"task_name": args.task_name,
				"stage": 1,
				"n_envs": args.n_envs,
				"noise_samples": args.noise_samples,
				"seed": args.seed,
				"n_seeds": args.n_seeds,
				"seeds": seeds,
				"ddim_steps": cfg.model.ddim_steps,
				"reward_threshold": reward_threshold,
				"exp_name": args.exp_name,
				"output_dir": args.out,
			},
			tags=[args.task_name, "lottery_ticket", "stage1"],
		)
	
	if not hasattr(cfg, 'device') or cfg.device is None:
		cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
	
	n_seeds = len(seeds)
	envs_per_seed = args.n_envs // n_seeds
	assert args.n_envs % n_seeds == 0, f"n_envs ({args.n_envs}) must be divisible by n_seeds ({n_seeds})"
	
	random.seed(seeds[0])
	torch.manual_seed(seeds[0])
	np.random.seed(seeds[0])
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = args.n_envs < 10
	
	all_fixed_seeds = []
	for seed in seeds:
		ss_env = np.random.SeedSequence(seed)
		fixed_seeds = [int(s.generate_state(1)[0]) for s in ss_env.spawn(envs_per_seed)]
		all_fixed_seeds.extend(fixed_seeds)
	
	env = build_lt_env(base_policy, cfg, args.n_envs, video_dir, save_vid=save_vid, fixed_seeds=all_fixed_seeds, reward_shaping=True)
	
	rngs = [np.random.default_rng(seed) for seed in seeds]
	
	seed_results = {seed: {"noise": [], "rewards": [], "success": [], "success_rates": [], "lengths": []} for seed in seeds}
	best_per_seed = {seed: {"success_rate": 0.0, "reward": 0.0, "episode_length": 0, "idx": -1, "noise": None} for seed in seeds}
	
	os.makedirs(args.out, exist_ok=True)
	
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	action_space_dim = env.action_space.shape[0]
	
	for noise_idx in range(args.noise_samples):
		print(f"Starting noise sample {noise_idx}")
		
		noise_vecs = []
		combined_noise_list = []
		for rng in rngs:
			noise_vec = rng.standard_normal(action_space_dim).astype(np.float32)
			# noise_vec = np.clip(noise_vec, env.action_space.low, env.action_space.high)
			noise_vecs.append(noise_vec)
			combined_noise_list.extend([noise_vec] * envs_per_seed)
		
		combined_noise = np.stack(combined_noise_list, axis=0)
		
		per_env_reward, per_env_success, per_env_length = evaluate_st1_noise(
			env, combined_noise, args.n_envs, save_vid, noise_idx, reward_threshold=reward_threshold
		)
		
		for seed_idx, seed in enumerate(seeds):
			start_idx = seed_idx * envs_per_seed
			end_idx = start_idx + envs_per_seed
			seed_results[seed]["noise"].append(noise_vecs[seed_idx])
			seed_results[seed]["rewards"].append(per_env_reward[start_idx:end_idx])
			seed_results[seed]["success"].append(per_env_success[start_idx:end_idx])
			seed_success_rate = float(np.mean(per_env_success[start_idx:end_idx]))
			seed_mean_reward = float(np.mean(per_env_reward[start_idx:end_idx]))
			seed_results[seed]["success_rates"].append(seed_success_rate)
			seed_results[seed]["lengths"].append(per_env_length[start_idx:end_idx])
			
			if (seed_success_rate > best_per_seed[seed]["success_rate"] or 
				(seed_success_rate == best_per_seed[seed]["success_rate"] and 
				 seed_mean_reward > best_per_seed[seed]["reward"])):
				best_per_seed[seed]["success_rate"] = seed_success_rate
				best_per_seed[seed]["reward"] = seed_mean_reward
				best_per_seed[seed]["episode_length"] = float(np.mean(per_env_length[start_idx:end_idx]))
				best_per_seed[seed]["idx"] = noise_idx
				best_per_seed[seed]["noise"] = noise_vecs[seed_idx].tolist()
		
		if not args.no_wandb:
			log_dict = {"iteration": noise_idx}
			for seed_idx, seed in enumerate(seeds):
				start_idx = seed_idx * envs_per_seed
				end_idx = start_idx + envs_per_seed
				log_dict[f"seed_{seed}/success_rate"] = float(np.mean(per_env_success[start_idx:end_idx]))
				log_dict[f"seed_{seed}/mean_reward"] = np.mean(per_env_reward[start_idx:end_idx])
				log_dict[f"seed_{seed}/std_reward"] = np.std(per_env_reward[start_idx:end_idx])
				log_dict[f"seed_{seed}/mean_episode_length"] = np.mean(per_env_length[start_idx:end_idx])
				log_dict[f"seed_{seed}/std_episode_length"] = np.std(per_env_length[start_idx:end_idx])
				log_dict[f"seed_{seed}/best_success_rate_so_far"] = best_per_seed[seed]["success_rate"]
				log_dict[f"seed_{seed}/best_idx_so_far"] = best_per_seed[seed]["idx"]
				log_dict[f"seed_{seed}/num_successes"] = sum(per_env_success[start_idx:end_idx])
			wandb.log(log_dict, step=noise_idx)
		
		if (noise_idx + 1) % 100 == 0:
			for seed_idx, seed in enumerate(seeds):
				save_results(args.out, seed_results[seed]["noise"], seed_results[seed]["rewards"], 
							seed_results[seed]["success"], seed_results[seed]["success_rates"], 
							seed_results[seed]["lengths"], seed, 
							all_fixed_seeds[seed_idx*envs_per_seed:(seed_idx+1)*envs_per_seed], 
							noise_idx=noise_idx, seed_id=seed)
			print(f"Saved checkpoint at sample {noise_idx + 1}")
	
	for seed_idx, seed in enumerate(seeds):
		save_results(args.out, seed_results[seed]["noise"], seed_results[seed]["rewards"], 
					seed_results[seed]["success"], seed_results[seed]["success_rates"], 
					seed_results[seed]["lengths"], seed, 
					all_fixed_seeds[seed_idx*envs_per_seed:(seed_idx+1)*envs_per_seed], 
					noise_idx=None, seed_id=seed)
	
	cross_seed_summary = {
		"seeds": seeds,
		"n_seeds": n_seeds,
		"envs_per_seed": envs_per_seed,
		"total_envs": args.n_envs,
		"noise_samples": args.noise_samples,
		"reward_threshold": reward_threshold,
		"per_seed_best": {}
	}
	
	for seed in seeds:
		cross_seed_summary["per_seed_best"][str(seed)] = {
			"best_success_rate": best_per_seed[seed]["success_rate"],
			"best_reward": best_per_seed[seed]["reward"],
			"best_episode_length": best_per_seed[seed]["episode_length"],
			"best_idx": best_per_seed[seed]["idx"],
			"best_noise": best_per_seed[seed]["noise"]
		}
	
	all_best_success_rates = [best_per_seed[seed]["success_rate"] for seed in seeds]
	cross_seed_summary["overall"] = {
		"mean_best_success_rate": float(np.mean(all_best_success_rates)),
		"std_best_success_rate": float(np.std(all_best_success_rates)),
		"max_best_success_rate": float(np.max(all_best_success_rates))
	}
	
	with open(os.path.join(args.out, "cross_seed_summary.json"), "w") as f:
		json.dump(cross_seed_summary, f, indent=2)
	
	env.close()
	
	if not args.no_wandb:
		final_summary = {
			"final/mean_best_success_rate": cross_seed_summary["overall"]["mean_best_success_rate"],
			"final/std_best_success_rate": cross_seed_summary["overall"]["std_best_success_rate"],
			"final/max_best_success_rate": cross_seed_summary["overall"]["max_best_success_rate"],
			"final/num_samples": args.noise_samples,
		}
		for seed in seeds:
			stats = cross_seed_summary["per_seed_best"][str(seed)]
			final_summary[f"final/seed_{seed}/best_success_rate"] = stats["best_success_rate"]
			final_summary[f"final/seed_{seed}/best_reward"] = stats["best_reward"]
			final_summary[f"final/seed_{seed}/best_episode_length"] = stats["best_episode_length"]
			final_summary[f"final/seed_{seed}/best_idx"] = stats["best_idx"]
		wandb.log(final_summary)
		wandb.finish()
	
	print(f"Search complete. Results saved to {args.out}")
	print(f"Best overall success rate: {cross_seed_summary['overall']['max_best_success_rate']:.4f}")

if __name__ == "__main__":
	main()
