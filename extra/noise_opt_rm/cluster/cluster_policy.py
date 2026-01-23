"""Evaluate cluster-based noise selection policy."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from noise_opt_rm.ticket_search.cluster.cluster_initial_obs import cluster_initial_obs

def get_clustering_dir_name(n_clusters, top_k=None):
	"""Generate clustering directory name with optional top_k suffix."""
	if top_k is not None:
		return f'clustering_k{n_clusters}_topk{top_k}'
	else:
		return f'clustering_k{n_clusters}'


def cluster_and_evaluate(result_dir, n_clusters=10, random_state=42, top_k=None):
	"""
	Cluster initial observations and evaluate cluster-based policy in one step.
	
	Args:
		result_dir: Path to lottery ticket results directory
		n_clusters: Number of clusters for K-means (default: 10)
		random_state: Random seed for reproducibility (default: 42)
		top_k: If provided, only use top k performing noises (default: None, use all)
	
	Returns:
		Dictionary with cluster policy evaluation results
	"""
	print(f"Clustering and evaluating policy from: {result_dir}")
	print(f"Using {n_clusters} clusters")
	if top_k is not None:
		print(f"Filtering to top {top_k} noises")
	
	# Use the existing cluster_initial_obs function
	obs_path = os.path.join(result_dir, 'initial_obs.npy')
	
	# Determine custom directory name if top_k is specified
	clustering_dir_name = get_clustering_dir_name(n_clusters, top_k)
	
	cluster_map = cluster_initial_obs(
		obs_path, 
		n_clusters=n_clusters, 
		random_state=random_state,
		output_path=result_dir,  # Pass result_dir as directory
		clustering_dir_name=clustering_dir_name
	)
	
	# Now evaluate the cluster policy
	print(f"\n{'='*60}")
	print("Evaluating cluster-based policy...")
	print(f"{'='*60}")
	
	# Load from the same directory where clustering was saved
	clustering_dir_name = get_clustering_dir_name(n_clusters, top_k)
	cluster_labels_path = os.path.join(result_dir, clustering_dir_name, 'cluster_labels.npy')
	results = evaluate_cluster_policy(result_dir, cluster_labels_path, n_clusters=n_clusters, top_k=top_k)
	
	return results


def evaluate_cluster_policy(result_dir, cluster_labels_path=None, n_clusters=None, top_k=None):
	"""
	Evaluate a policy that selects the best noise for each cluster.
	
	For each cluster, we find the noise sample that performs best on that cluster's
	initial observations, then compute the overall success rate by assigning each
	observation to its cluster's best noise.
	
	Args:
		result_dir: Path to lottery ticket results directory containing:
			- successes.npy: (n_noise_samples, n_envs) array of success flags
			- initial_obs.npy: (n_envs, obs_dim) array of initial observations
			- clustering/cluster_labels.npy: (n_envs,) array of cluster assignments
			- summary.json: metadata about the search
		cluster_labels_path: Optional path to cluster_labels.npy (default: result_dir/clustering/cluster_labels.npy)
		n_clusters: Number of clusters (used for output directory naming)
		top_k: If provided, only use top k performing noises (default: None, use all)
	
	Returns:
		Dictionary with cluster policy evaluation results
	"""
	print(f"Evaluating cluster-based policy from: {result_dir}")
	
	# Load data
	successes = np.load(os.path.join(result_dir, 'successes.npy'), allow_pickle=True)
	initial_obs = np.load(os.path.join(result_dir, 'initial_obs.npy'))
	
	with open(os.path.join(result_dir, 'summary.json'), 'r') as f:
		summary = json.load(f)
	
	# Load cluster labels
	if cluster_labels_path is None:
		cluster_labels_path = os.path.join(result_dir, 'clustering_k*', 'cluster_labels.npy')
		# Find the clustering directory
		import glob
		matching_paths = glob.glob(cluster_labels_path)
		if matching_paths:
			cluster_labels_path = matching_paths[0]  # Use the first match
		else:
			raise FileNotFoundError(f"No clustering found in: {result_dir}")
	
	if not os.path.exists(cluster_labels_path):
		raise FileNotFoundError(f"Cluster labels not found at: {cluster_labels_path}")
	
	cluster_labels = np.load(cluster_labels_path)
	n_envs, obs_dim = initial_obs.shape
	n_noise_samples_total = len(successes)
	n_clusters = len(np.unique(cluster_labels))
	
	# Filter to top k noises if specified
	if top_k is not None:
		# Get indices of top k noises by success rate
		success_rates = np.array([successes[i].mean() for i in range(n_noise_samples_total)])
		top_k_indices = np.argsort(success_rates)[::-1][:top_k]  # Descending order, take top k
		print(f"\nFiltering to top {top_k} noises (indices: {top_k_indices[:10]}...)")
		print(f"  Success rates range: {success_rates[top_k_indices].min():.2%} - {success_rates[top_k_indices].max():.2%}")
		
		# Filter successes to only top k noises
		successes = successes[top_k_indices]
		n_noise_samples = len(successes)
		
		# Create a mapping from filtered index to original index
		filtered_to_original = {i: int(top_k_indices[i]) for i in range(len(top_k_indices))}
	else:
		n_noise_samples = n_noise_samples_total
		filtered_to_original = {i: i for i in range(n_noise_samples)}
	
	print(f"\nData loaded:")
	print(f"  {n_envs} environments, {obs_dim}D observations")
	print(f"  {n_noise_samples} noise samples (total: {n_noise_samples_total})")
	print(f"  {n_clusters} clusters")
	
	# For each cluster, find the best noise (highest success rate on that cluster)
	cluster_best_noise = {}  # cluster_id -> (noise_idx, success_rate)
	cluster_noise_success_rates = {}  # cluster_id -> {noise_idx: success_rate}
	
	for cluster_id in range(n_clusters):
		cluster_mask = cluster_labels == cluster_id
		cluster_size = cluster_mask.sum()
		
		if cluster_size == 0:
			print(f"Warning: Cluster {cluster_id} has no observations, skipping")
			continue
		
		# Evaluate each noise sample on this cluster
		noise_success_rates = {}
		for noise_idx in range(n_noise_samples):
			# Get successes for this noise on this cluster's environments
			cluster_successes = successes[noise_idx][cluster_mask]
			success_rate = cluster_successes.mean()
			noise_success_rates[noise_idx] = success_rate
		
		cluster_noise_success_rates[cluster_id] = noise_success_rates
		
		# Find best noise for this cluster
		best_filtered_idx, best_sr = max(noise_success_rates.items(), key=lambda x: x[1])
		best_original_idx = filtered_to_original[best_filtered_idx]
		cluster_best_noise[cluster_id] = (best_filtered_idx, best_sr, best_original_idx)
		
		print(f"  Cluster {cluster_id} ({cluster_size} envs): Best noise = {best_original_idx} (filtered idx: {best_filtered_idx}, success rate: {best_sr:.2%})")
	
	# Compute overall success rate using cluster-based policy
	# For each environment, use the best noise for its cluster
	cluster_policy_successes = []
	for env_idx in range(n_envs):
		cluster_id = cluster_labels[env_idx]
		if cluster_id not in cluster_best_noise:
			print(f"Warning: Environment {env_idx} in cluster {cluster_id} with no best noise, marking as failure")
			cluster_policy_successes.append(False)
			continue
		
		best_filtered_idx = cluster_best_noise[cluster_id][0]
		success = successes[best_filtered_idx][env_idx]
		cluster_policy_successes.append(bool(success))
	
	cluster_policy_success_rate = np.mean(cluster_policy_successes)
	
	# Compare with baseline policies
	baseline_success_rate = summary['success_rates_sorted'][0]  # Best single noise
	random_success_rate = np.mean([sr for sr in summary['success_rates_sorted']])  # Average noise
	
	print(f"\n=== Policy Comparison ===")
	print(f"Cluster-based policy: {cluster_policy_success_rate:.2%}")
	print(f"Best single noise:    {baseline_success_rate:.2%}")
	print(f"Random noise (avg):   {random_success_rate:.2%}")
	print(f"Improvement over best: {(cluster_policy_success_rate - baseline_success_rate):.2%}")
	
	# Analyze per-cluster improvements
	cluster_improvements = {}
	
	# Get the globally best noise index from ranking
	# The ranking.txt has noise indices in sorted order, or we can use best_original_index
	global_best_noise_idx = summary.get('best_original_index', 0)
	
	# For top_k filtering, need to check if global best is in filtered set
	if top_k is not None:
		# Find the filtered index of global best, if it exists
		reverse_mapping = {v: k for k, v in filtered_to_original.items()}
		if global_best_noise_idx in reverse_mapping:
			global_best_filtered_idx = reverse_mapping[global_best_noise_idx]
		else:
			# Global best not in top k, use the best from top k
			global_best_filtered_idx = 0  # Best in filtered set
			global_best_noise_idx = filtered_to_original[0]
			print(f"\nNote: Global best noise #{global_best_noise_idx} not in top {top_k}, using #{global_best_noise_idx} instead")
	else:
		global_best_filtered_idx = global_best_noise_idx
	
	for cluster_id, (best_filtered_idx, cluster_sr, best_original_idx) in cluster_best_noise.items():
		cluster_mask = cluster_labels == cluster_id
		# Compare to what the globally best noise achieves on this cluster
		global_best_on_cluster = successes[global_best_filtered_idx][cluster_mask].mean()
		improvement = cluster_sr - global_best_on_cluster
		cluster_improvements[cluster_id] = {
			'best_noise_idx': int(best_original_idx),
			'best_filtered_idx': int(best_filtered_idx),
			'cluster_success_rate': float(cluster_sr),
			'global_best_on_cluster': float(global_best_on_cluster),
			'improvement': float(improvement),
			'cluster_size': int(cluster_mask.sum())
		}
	
	# Save results
	results = {
		'cluster_policy_success_rate': float(cluster_policy_success_rate),
		'baseline_success_rate': float(baseline_success_rate),
		'random_success_rate': float(random_success_rate),
		'improvement_over_best': float(cluster_policy_success_rate - baseline_success_rate),
		'n_clusters': int(n_clusters),
		'n_envs': int(n_envs),
		'n_noise_samples': int(n_noise_samples),
		'n_noise_samples_total': int(n_noise_samples_total),
		'top_k': top_k,
		'cluster_best_noise': {int(k): {'noise_idx': int(v[2]), 'filtered_idx': int(v[0]), 'success_rate': float(v[1])} 
		                       for k, v in cluster_best_noise.items()},
		'cluster_improvements': cluster_improvements,
		'cluster_policy_successes': [bool(s) for s in cluster_policy_successes],
		'filtered_to_original_mapping': filtered_to_original,
	}
	
	# Save to clustering directory
	clustering_dir = os.path.dirname(cluster_labels_path)
	output_dir = clustering_dir
	os.makedirs(output_dir, exist_ok=True)
	
	output_path = os.path.join(output_dir, 'cluster_policy_eval.json')
	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)
	print(f"\nSaved cluster policy evaluation to: {output_path}")
	
	# Print per-cluster analysis
	print(f"\n=== Per-Cluster Analysis ===")
	for cluster_id in sorted(cluster_improvements.keys()):
		info = cluster_improvements[cluster_id]
		print(f"Cluster {cluster_id} ({info['cluster_size']} envs):")
		print(f"  Best noise: #{info['best_noise_idx']} -> {info['cluster_success_rate']:.2%}")
		print(f"  Global best on this cluster: {info['global_best_on_cluster']:.2%}")
		print(f"  Improvement: {info['improvement']:+.2%}")
	
	return results


def main():
	parser = argparse.ArgumentParser(description='Evaluate cluster-based noise selection policy')
	parser.add_argument('result_dir', type=str, help='Path to lottery ticket results directory')
	parser.add_argument('--cluster_labels', type=str, default=None, 
	                   help='Path to cluster_labels.npy (default: result_dir/clustering/cluster_labels.npy)')
	parser.add_argument('--n_clusters', type=int, default=None,
	                   help='Number of clusters to create (if not already clustered)')
	parser.add_argument('--top_k', type=int, default=None,
	                   help='Only use top k performing noises for cluster policy (default: None, use all)')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for clustering (default: 42)')
	
	args = parser.parse_args()
	
	if not os.path.exists(args.result_dir):
		raise FileNotFoundError(f"Result directory not found: {args.result_dir}")
	
	# Check if we need to cluster first
	if args.cluster_labels:
		cluster_labels_path = args.cluster_labels
	elif args.n_clusters is not None:
		cluster_labels_path = None  # Will be created by cluster_and_evaluate
	else:
		# Look for existing clustering
		import glob
		pattern = os.path.join(args.result_dir, 'clustering_k*', 'cluster_labels.npy')
		matching_paths = glob.glob(pattern)
		cluster_labels_path = matching_paths[0] if matching_paths else None
	
	if args.n_clusters is not None:
		# User wants to create new clustering
		print(f"Creating new clustering with {args.n_clusters} clusters...")
		results = cluster_and_evaluate(args.result_dir, args.n_clusters, args.seed, args.top_k)
	elif cluster_labels_path is None:
		# No clustering exists, create default
		print(f"No existing clustering found, creating with default 10 clusters...")
		results = cluster_and_evaluate(args.result_dir, n_clusters=10, random_state=args.seed, top_k=args.top_k)
	else:
		# Use existing clustering
		print(f"Using existing clustering from: {cluster_labels_path}")
		# Extract n_clusters from path if needed
		if args.n_clusters is None:
			import re
			match = re.search(r'clustering_k(\d+)', cluster_labels_path)
			extracted_n_clusters = int(match.group(1)) if match else None
		else:
			extracted_n_clusters = args.n_clusters
		results = evaluate_cluster_policy(args.result_dir, cluster_labels_path, n_clusters=extracted_n_clusters, top_k=args.top_k)
	
	print(f"\n{'='*60}")
	print(f"Cluster Policy Success Rate: {results['cluster_policy_success_rate']:.2%}")
	print(f"Improvement over best single noise: {results['improvement_over_best']:+.2%}")
	print(f"{'='*60}")


if __name__ == '__main__':
	main()
