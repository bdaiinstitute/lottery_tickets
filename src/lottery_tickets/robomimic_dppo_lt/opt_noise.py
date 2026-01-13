# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
Eval for noise search
-------------------------
Notes on seeding in envs:
-------------------------
Wrapper order: MujocoEnv -> RobotEnv -> ManipulatorEnv -> PickPlace (Robosuite) -> RobosuiteEnv (Robomimic) ->
ObservationWrapperRobomimic (DSRL, parent: gym.Env) -> ActionChunkWrapper (p: gym.Env) -> VecEnv (SB3) -> DiffusionPolicyEnvWrapper (p: VecEnvWrapper- SB3)

- The whole Robosuite and Robomimic stack does not set or use seeds or generators upto 1.4.1 (currently used by dsrl)
- However, the env wrappers enable the flow of seed till MujocoEnv which sets a self.rng in its init in 1.5.1 (used by vpl)- which is then used for sampling initial states
- ActionChunkWrapper passes up the seed in reset to ObservationWrapperRobomimic, which does seed np.random
- SB3 VecEnv defines a _seed which stores a list fo seeds for each env. This is initialized to None, and can be set using seed()- to ensure heterogeneous envs
- reset in SB3 SubprocVecEnv sends the respective seed to each env's FIRST reset call. SB3 DummyVecEnv does the same.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["MUJOCO_GL"] = "egl"

BASE_DIR = Path(__file__).resolve().parents[2]
if BASE_DIR.as_posix() not in sys.path:
	sys.path.append(BASE_DIR.as_posix())

from env_util import build_single_env
from eval_utils import evaluate_noise_single, load_noise_idx, save_eval_serial
from policy_util import load_base_policy

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def parse_args() -> argparse.Namespace:
	"""Helper function to parse arguments.
	
	Returns:
		argparse.Namespace: Parsed arguments.
	"""
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/noise_eval_results/")
	p.add_argument("--eval", type=str, default=None)
	p.add_argument("--eval_idx", type=int, nargs="+", default=[0], help="List of noise indices to evaluate")
	p.add_argument("--save_vid", action="store_true", help="Save evaluation videos")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--epsilon", type=float, default=None, help="DDIM steps to override config value")
	return p.parse_args()

def main():
	"""Main function for noise evaluation."""
	args = parse_args()
	base_path = BASE_DIR.as_posix()
	config_path = os.path.join(f"{base_path}/lottery_tickets/robomimic_dppo_lt", TASK_CONFIGS[args.task_name])

	OmegaConf.register_new_resolver("eval", eval)
	config_dir = os.path.dirname(config_path)
	config_name = os.path.basename(config_path).replace(".yaml", "")
	
	with initialize_config_dir(version_base=None, config_dir=config_dir):
		cfg = compose(config_name=config_name)
	OmegaConf.set_struct(cfg, False)
	cfg.seed = args.seed
	if not hasattr(cfg, "device") or cfg.device is None:
		cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
	
	# Override ddim_steps if provided
	if args.ddim_steps is not None:
		cfg.model.ddim_steps = args.ddim_steps
	
	# Extract ticket name from eval path (last folder in the path)
	ticket_name = os.path.basename(args.eval.rstrip("/")) if args.eval else None
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	if ticket_name:
		if len(args.eval_idx) > 1:
			idx_str = f"idx{min(args.eval_idx)}-{max(args.eval_idx)}_n{len(args.eval_idx)}"
		else:
			idx_str = f"idx{args.eval_idx[0]}"
		base_out = os.path.join(args.out, args.task_name, ticket_name, f"eval_{idx_str}_seed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	else:
		if len(args.eval_idx) > 1:
			idx_str = f"idx{min(args.eval_idx)}-{max(args.eval_idx)}_n{len(args.eval_idx)}"
		else:
			idx_str = f"idx{args.eval_idx[0]}"
		base_out = os.path.join(args.out, args.task_name, f"eval_{idx_str}_seed{args.seed}_ddim{cfg.model.ddim_steps}_{timestamp}")
	
	os.makedirs(base_out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(base_out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(base_out, "raw_videos")
	save_vid = args.save_vid
	
	# Track results across all eval indices
	all_indices_results = {}
	
	# Loop over all eval indices
	for eval_idx in args.eval_idx:
		print(f"\n{'='*80}")
		print(f"Evaluating noise index: {eval_idx} ({args.eval_idx.index(eval_idx) + 1}/{len(args.eval_idx)})")
		print(f"{'='*80}")
		
		random.seed(cfg.seed)
		np.random.seed(cfg.seed)
		torch.manual_seed(cfg.seed)	

		best_noise = load_noise_idx(args.eval, eval_idx)
		print(f">>> Loaded noise idx {eval_idx} from {args.eval}")
		
		# Create subdirectory for this eval_idx
		args.out = os.path.join(base_out, f"noise_idx_{eval_idx}")
		os.makedirs(args.out, exist_ok=True)
		
		# Prepare noise vector once
		noise_vec = best_noise.astype(np.float32).flatten()
		
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
			# noise_vec_clipped = np.clip(noise_vec, env.action_space.low, env.action_space.high)
			
			# Collect results for this seed across n_evals_per_seed
			seed_rewards = []
			seed_successes = []
			
			for eval_iter in range(args.n_evals_per_seed):
				print(f"  Evaluation {eval_iter + 1}/{args.n_evals_per_seed}")
				if (epsilon := args.epsilon) is not None:
					if torch.rand(()) < epsilon:
						episode_reward, success = evaluate_noise_single(
							env, torch.randn(noise_vec.shape), save_vid, noise_idx=eval_idx, 
							eval_num=eval_iter, rew_offset=cfg.env.reward_offset
						)
					else:
						episode_reward, success = evaluate_noise_single(
							env, noise_vec, save_vid, noise_idx=eval_idx, 
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
			args.out, rewards_all, success_flags_all, episode_seeds, 
			ticket_name=ticket_name,
			eval_noise_idx=eval_idx,
			eval_seed=args.seed
		)
		print(
			f"Noise {eval_idx} complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
			f"Success mean={success_mean:.4f} std={success_std:.4f}"
		)
		
		# Store results for this index
		all_indices_results[eval_idx] = {
			"reward_mean": float(reward_mean),
			"reward_std": float(reward_std),
			"success_mean": float(success_mean),
			"success_std": float(success_std)
		}
	
	# Compute aggregate statistics across all eval indices
	if len(args.eval_idx) > 1:
		reward_means = [all_indices_results[idx]["reward_mean"] for idx in args.eval_idx]
		success_means = [all_indices_results[idx]["success_mean"] for idx in args.eval_idx]
		
		aggregate_stats = {
			"n_noise_indices": len(args.eval_idx),
			"eval_indices": args.eval_idx,
			"reward_mean_across_indices": float(np.mean(reward_means)),
			"reward_std_across_indices": float(np.std(reward_means, ddof=1 if len(reward_means) > 1 else 0)),
			"success_mean_across_indices": float(np.mean(success_means)),
			"success_std_across_indices": float(np.std(success_means, ddof=1 if len(success_means) > 1 else 0)),
			"per_index_results": all_indices_results
		}
		
		# Save aggregate results
		with open(os.path.join(base_out, "aggregate_results.json"), "w") as f:
			json.dump(aggregate_stats, f, indent=2)
		
		print(f"\n{'='*80}")
		print("AGGREGATE RESULTS ACROSS ALL NOISE INDICES")
		print(f"{'='*80}")
		print(f"Number of noise indices evaluated: {len(args.eval_idx)}")
		print(f"Reward mean across indices: {aggregate_stats['reward_mean_across_indices']:.4f} ± {aggregate_stats['reward_std_across_indices']:.4f}")
		print(f"Success mean across indices: {aggregate_stats['success_mean_across_indices']:.4f} ± {aggregate_stats['success_std_across_indices']:.4f}")
		print(f"Results saved to: {base_out}/aggregate_results.json")
		print(f"{'='*80}")

if __name__ == "__main__":
	main()
