"""
Eval for cluster-based noise policy
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
if str(BASE_DIR) not in sys.path:
	sys.path.append(str(BASE_DIR))

from noise_opt_rm import build_single_env, load_base_policy
from noise_opt_rm.eval.eval_utils import evaluate_noise_single, save_eval_serial

TASK_CONFIGS = {
	"lift": "cfg/robomimic/dsrl_lift.yaml",
	"can": "cfg/robomimic/dsrl_can.yaml",
	"square": "cfg/robomimic/dsrl_square.yaml",
	"transport": "cfg/robomimic/dsrl_transport.yaml",
}

def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--task_name", default="lift", choices=list(TASK_CONFIGS.keys()))
	p.add_argument("--n_evals_per_seed", type=int, default=100)
	p.add_argument("--n_seeds", type=int, default=5, help="Number of evaluation seeds")
	p.add_argument("--seed", type=int, default=1619, help="Random seed for environment")
	p.add_argument("--out", default="logs_res_rm/cluster_eval_results/")
	p.add_argument("--ticket_dir", type=str, required=True, help="Path to lottery ticket results directory")
	p.add_argument("--n_clusters", type=int, default=None, help="Number of clusters to use (default: use first found)")
	p.add_argument("--top_k", type=int, default=None, help="If specified, use cluster policy trained on top k noises")
	p.add_argument("--save_vid", type=int, default=False)
	return p.parse_args()

def get_eval_output_dir_name(ticket_name, n_clusters, seed, timestamp, top_k=None):
	"""Generate evaluation output directory name with optional top_k suffix."""
	if top_k is not None:
		return f"{ticket_name}_cluster_k{n_clusters}_topk{top_k}_seed{seed}_{timestamp}"
	else:
		return f"{ticket_name}_cluster_k{n_clusters}_seed{seed}_{timestamp}"


def get_clustering_dir_name(n_clusters, top_k=None):
	"""Generate clustering directory name with optional top_k suffix."""
	if top_k is not None:
		return f'clustering_k{n_clusters}_topk{top_k}'
	else:
		return f'clustering_k{n_clusters}'


def load_cluster_policy(ticket_dir, n_clusters=None, top_k=None):
	"""Load cluster policy from lottery ticket results."""
	# Look for cluster policy evaluation results (try all clustering_k* directories)
	import glob
	
	if n_clusters is not None:
		# Look for specific k value (with or without top_k suffix)
		clustering_dir_name = get_clustering_dir_name(n_clusters, top_k)
		pattern = os.path.join(ticket_dir, clustering_dir_name, 'cluster_policy_eval.json')
		matching_paths = glob.glob(pattern)
		if not matching_paths:
			topk_str = f" with top_k={top_k}" if top_k is not None else ""
			raise FileNotFoundError(
				f"Cluster policy with k={n_clusters}{topk_str} not found in: {ticket_dir}\n"
				f"Run: python noise_opt_rm/noise_search/cluster_policy.py {ticket_dir} --n_clusters {n_clusters}"
			)
	else:
		# Use any available clustering
		pattern = os.path.join(ticket_dir, 'clustering_k*', 'cluster_policy_eval.json')
		matching_paths = glob.glob(pattern)
		if not matching_paths:
			raise FileNotFoundError(
				f"Cluster policy not found in: {ticket_dir}\n"
				f"Run cluster_policy.py first to generate the policy."
			)
	
	# Use the first (or only) match
	cluster_policy_path = matching_paths[0]
	clustering_dir = os.path.dirname(cluster_policy_path)
	
	print(f"Loading cluster policy from: {cluster_policy_path}")
	
	with open(cluster_policy_path, 'r') as f:
		cluster_policy = json.load(f)
	
	# Load cluster labels and noise samples
	cluster_labels = np.load(os.path.join(clustering_dir, 'cluster_labels.npy'))
	noise_samples = np.load(os.path.join(ticket_dir, 'noise_samples.npy'))
	
	# Load KMeans model for assigning new observations to clusters
	import pickle
	kmeans_path = os.path.join(clustering_dir, 'kmeans.pkl')
	with open(kmeans_path, 'rb') as f:
		kmeans = pickle.load(f)
	
	return cluster_policy, kmeans, noise_samples

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
	
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	
	# Load cluster policy
	cluster_policy, kmeans, noise_samples = load_cluster_policy(args.ticket_dir, args.n_clusters, args.top_k)
	print(f">>> Loaded cluster policy from {args.ticket_dir}")
	print(f"    {cluster_policy['n_clusters']} clusters, {cluster_policy['n_noise_samples']} noise samples")
	if cluster_policy.get('top_k') is not None:
		print(f"    Using top {cluster_policy['top_k']} noises (total: {cluster_policy['n_noise_samples_total']})")
	print(f"    Train success rate: {cluster_policy['cluster_policy_success_rate']:.2%}")
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	ticket_name = os.path.basename(args.ticket_dir.rstrip('/'))
	n_clusters_used = cluster_policy['n_clusters']
	top_k_used = cluster_policy.get('top_k')
	
	dir_name = get_eval_output_dir_name(ticket_name, n_clusters_used, args.seed, timestamp, top_k_used)
	args.out = os.path.join(args.out, args.task_name, dir_name)
	os.makedirs(args.out, exist_ok=True)
	
	# Dump resolved config
	with open(os.path.join(args.out, "config.yaml"), "w") as f:
		f.write(OmegaConf.to_yaml(cfg))
	
	base_policy = load_base_policy(cfg)
	video_dir = os.path.join(args.out, "raw_videos")
	save_vid = args.save_vid
	
	rewards_all = []
	success_flags_all = []
	episode_seeds = []
	cluster_assignments = []  # Track which cluster each env was assigned to
	noise_indices_used = []   # Track which noise was used for each env

	# Serial rollouts: iterate over seeds, then evals per seed
	for seed_idx in range(args.n_seeds):
		current_seed = args.seed + seed_idx
		episode_seeds.append(current_seed)
		print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds}: {current_seed} ===")
		
		# Build new environment for this seed
		env = build_single_env(base_policy, cfg, video_dir, current_seed, save_vid=save_vid)
		
		# Collect results for this seed across n_evals_per_seed
		seed_rewards = []
		seed_successes = []
		seed_clusters = []
		seed_noises = []
		
		for eval_idx in range(args.n_evals_per_seed):
			print(f"  Evaluation {eval_idx + 1}/{args.n_evals_per_seed}")
			
			# Get initial observation
			obs = env.reset()
			
			# Assign to cluster using KMeans
			obs_flat = obs.flatten().reshape(1, -1)
			cluster_id = int(kmeans.predict(obs_flat)[0])
			
			# Get the best noise for this cluster
			noise_idx = cluster_policy['cluster_best_noise'][str(cluster_id)]['noise_idx']
			noise_vec = noise_samples[noise_idx].astype(np.float32).flatten()
			# noise_vec_clipped = np.clip(noise_vec, env.action_space.low, env.action_space.high)
			noise_vec_clipped = noise_vec

			# Evaluate with this noise
			episode_reward, success = evaluate_noise_single(
				env, noise_vec_clipped, save_vid, noise_idx=noise_idx, 
				eval_num=eval_idx, rew_offset=cfg.env.reward_offset, initial_obs=obs
			)
			
			seed_rewards.append(float(episode_reward))
			seed_successes.append(bool(success))
			seed_clusters.append(int(cluster_id))
			seed_noises.append(int(noise_idx))
			print(f"    Cluster: {cluster_id}, Noise: {noise_idx}, Reward: {episode_reward:.4f}, Success: {success}")
		
		rewards_all.append(seed_rewards)
		success_flags_all.append(seed_successes)
		cluster_assignments.append(seed_clusters)
		noise_indices_used.append(seed_noises)
		env.close()
		print(f"  Seed {current_seed} complete - Mean reward: {np.mean(seed_rewards):.4f}, Success rate: {np.mean(seed_successes):.4f}")

	reward_mean, reward_std, success_mean, success_std = save_eval_serial(
		args.out, rewards_all, success_flags_all, episode_seeds, 
		ticket_name=ticket_name,
		eval_seed=args.seed,
		cluster_assignments=cluster_assignments,
		noise_indices_used=noise_indices_used,
		cluster_policy_train_success=cluster_policy['cluster_policy_success_rate']
	)
	
	print(f"\n{'='*60}")
	print(f"Cluster Policy Evaluation Complete")
	print(f"Train success rate: {cluster_policy['cluster_policy_success_rate']:.2%}")
	print(f"Test success rate:  {success_mean:.2%} ± {success_std:.2%}")
	print(f"Test reward:        {reward_mean:.4f} ± {reward_std:.4f}")
	print(f"{'='*60}")

if __name__ == "__main__":
	main()
