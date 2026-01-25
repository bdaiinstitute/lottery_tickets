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

os.environ["MUJOCO_GL"] = "egl"

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_single_env, load_base_policy
from noise_opt_rm.eval.eval_utils import save_eval_serial, load_noise_idx

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
	p.add_argument("--n_seeds", type=int, default=1)
	p.add_argument("--seed", type=int, default=202020, help="Starting seed")
	p.add_argument("--out", default="logs_res_rm/policy_record/")
	p.add_argument("--ddim_steps", type=int, default=None, help="DDIM steps to override config value")
	p.add_argument("--noise_path", type=str, default=None, help="Path to noise file to load")
	p.add_argument("--noise_idx", type=int, default=0, help="Index of noise to load from file")
	return p.parse_args()

def _resolve_out(out_path: str, task_name: str, n_seeds: int, n_evals_per_seed: int, seed: int, ddim_steps: int, noise_path: str = None, noise_idx: int = None) -> str:
	ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
	if noise_path is not None:
		noise_name = os.path.basename(noise_path.rstrip('/'))
		run_name = f"{noise_name}_idx{noise_idx}_seeds{n_seeds}_evals{n_evals_per_seed}_seed{seed}_ddim{ddim_steps}_{ts}"
	else:
		run_name = f"base_policy_seeds{n_seeds}_evals{n_evals_per_seed}_seed{seed}_ddim{ddim_steps}_{ts}"
	return os.path.join(out_path.rstrip('/'), task_name, run_name)

def evaluate_policy_single(env, save_vid=False, eval_num=0, rew_offset=0.0, noise_vec=None):
	if save_vid:
		env.env.name_prefix = f"ep_{eval_num}"
	
	env.reset()
	episode_reward = 0.0
	success = False
	done = False

	action = noise_vec.reshape(1, -1).astype(np.float32)
	steps = 0
	while not done:
		_, r, d, info = env.step(action)
		steps += 1
		r_val = float(r[0])
		episode_reward += r_val
		
		if r_val > float(-rew_offset):
			success = True
		
		if d[0]:
			done = True
			
	return episode_reward, success

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

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	
	args.out = _resolve_out(args.out, args.task_name, args.n_seeds, args.n_evals_per_seed, args.seed, cfg.model.ddim_steps, args.noise_path, args.noise_idx)
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	

	video_dir = os.path.join(args.out, "raw_videos")
	record_dir = os.path.join(args.out, "recorded_episodes")
	save_vid = True
	
	# Load noise if provided
	noise_vec=None
	cfg.model.controllable_noise = False # IMP
	if args.noise_path is not None:
		cfg.model.controllable_noise = True # IMP
		noise_vec = load_noise_idx(args.noise_path, args.noise_idx).astype(np.float32).flatten()
		print(f">>> Loaded noise idx {args.noise_idx} from {args.noise_path}")
	
	base_policy = load_base_policy(cfg)
	rewards_all = []
	success_flags_all = []
	episode_seeds = []

	for seed_idx in range(args.n_seeds):
		current_seed = args.seed + seed_idx
		episode_seeds.append(current_seed)
		print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
		
		env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid, record_dir=record_dir)
		if not args.noise_path:
			action_space_dim = env.action_space.shape[0]
			noise_vec = np.zeros((1, action_space_dim), dtype=np.float32) # controllability noise disabled

		seed_rewards = []
		seed_successes = []
		
		for eval_idx in range(args.n_evals_per_seed):
			print(f"  Evaluation {eval_idx + 1}/{args.n_evals_per_seed}")
			episode_reward, success = evaluate_policy_single(
				env, save_vid, eval_num=eval_idx, rew_offset=cfg.env.reward_offset, noise_vec=noise_vec
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
		eval_type="base_policy",
		eval_seed=args.seed,
		n_seeds=args.n_seeds,
		n_evals_per_seed=args.n_evals_per_seed
	)
	print(
		f"Evaluation complete. Reward mean={reward_mean:.4f} std={reward_std:.4f}; "
		f"Success mean={success_mean:.4f} std={success_std:.4f}"
	)

if __name__ == "__main__":
	main()
