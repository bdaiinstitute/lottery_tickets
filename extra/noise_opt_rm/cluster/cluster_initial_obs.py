"""Cluster initial observations using K-means."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_initial_obs(obs_path, n_clusters=5, output_path=None, random_state=42, clustering_dir_name=None):
	"""
	Cluster high-dimensional initial observations using K-means.
	
	Args:
		obs_path: Path to initial_obs.npy file with shape (n_envs, obs_dim)
		n_clusters: Number of clusters (default: 5)
		output_path: Path to save clustering results (default: same dir as obs_path)
		random_state: Random seed for reproducibility (default: 42)
		clustering_dir_name: Custom name for clustering directory (default: clustering_k{n_clusters})
	
	Returns:
		Dictionary mapping cluster_id to list of observation indices
	"""
	# Load initial observations
	initial_obs = np.load(obs_path)
	print(f"Loaded initial observations with shape: {initial_obs.shape}")
	n_envs, obs_dim = initial_obs.shape
	
	# Validate n_clusters
	if n_clusters > n_envs:
		print(f"Warning: n_clusters ({n_clusters}) > n_envs ({n_envs}), setting n_clusters={n_envs}")
		n_clusters = n_envs
	
	print(f"Running K-means with n_clusters={n_clusters}...")
	
	# Run K-means clustering
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
	cluster_labels = kmeans.fit_predict(initial_obs)
	
	# Calculate silhouette score if we have more than 1 cluster
	silhouette = None
	if n_clusters > 1 and n_envs > n_clusters:
		try:
			silhouette = float(silhouette_score(initial_obs, cluster_labels))
			print(f"Silhouette score: {silhouette:.4f}")
		except Exception as e:
			print(f"Could not calculate silhouette score: {e}")
	
	# Count clusters
	unique_labels = np.unique(cluster_labels)
	n_clusters_found = len(unique_labels)
	
	print(f"\nClustering Results:")
	print(f"  Found {n_clusters_found} clusters")
	print(f"  Cluster labels: {unique_labels.tolist()}")
	
	# Create cluster mapping: cluster_id -> [obs_idx1, obs_idx2, ...]
	cluster_map = {}
	for cluster_id in unique_labels:
		obs_indices = np.where(cluster_labels == cluster_id)[0].tolist()
		cluster_map[int(cluster_id)] = obs_indices
		print(f"  Cluster {cluster_id}: {len(obs_indices)} observations")
	
	# Determine output directory
	if output_path is None:
		output_dir = os.path.dirname(obs_path)
	else:
		output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else os.path.dirname(obs_path)
	
	# Create clustering subdirectory with custom name or default
	if clustering_dir_name is None:
		clustering_dir_name = f'clustering_k{n_clusters}'
	clustering_dir = os.path.join(output_dir, clustering_dir_name)
	os.makedirs(clustering_dir, exist_ok=True)
	print(f"\nSaving clustering results to: {clustering_dir}")
	
	# Save cluster mapping as JSON
	cluster_map_path = os.path.join(clustering_dir, 'cluster_map.json')
	with open(cluster_map_path, 'w') as f:
		json.dump(cluster_map, f, indent=2)
	print(f"Saved cluster mapping to: {cluster_map_path}")
	
	# Save cluster labels as numpy array
	labels_path = os.path.join(clustering_dir, 'cluster_labels.npy')
	np.save(labels_path, cluster_labels)
	print(f"Saved cluster labels to: {labels_path}")
	
	# Save cluster centers
	centers_path = os.path.join(clustering_dir, 'cluster_centers.npy')
	np.save(centers_path, kmeans.cluster_centers_)
	print(f"Saved cluster centers to: {centers_path}")
	
	# Save clustering statistics
	stats = {
		"n_observations": int(n_envs),
		"n_clusters": int(n_clusters_found),
		"cluster_sizes": {int(k): len(v) for k, v in cluster_map.items()},
		"silhouette_score": silhouette,
		"inertia": float(kmeans.inertia_),
	}
	
	stats_path = os.path.join(clustering_dir, 'clustering_stats.json')
	with open(stats_path, 'w') as f:
		json.dump(stats, f, indent=2)
	print(f"Saved clustering statistics to: {stats_path}")
	
	# Save kmeans object for later use
	try:
		kmeans_path = os.path.join(clustering_dir, 'kmeans.pkl')
		with open(kmeans_path, 'wb') as f:
			pickle.dump(kmeans, f)
		print(f"Saved K-means object to: {kmeans_path}")
	except Exception as e:
		print(f"Warning: Could not save K-means object: {e}")
	
	return cluster_map


def main():
	parser = argparse.ArgumentParser(description='Cluster initial observations using K-means')
	parser.add_argument('obs_path', type=str, help='Path to initial_obs.npy file')
	parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters (default: 5)')
	parser.add_argument('--output', type=str, default=None, help='Output path for clustering results (default: same dir as input)')
	parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
	
	args = parser.parse_args()
	
	if not os.path.exists(args.obs_path):
		raise FileNotFoundError(f"File not found: {args.obs_path}")
	
	cluster_map = cluster_initial_obs(
		args.obs_path,
		args.n_clusters,
		args.output,
		args.seed
	)
	
	print(f"\nClustering complete!")
	print(f"Cluster summary:")
	for cluster_id, obs_indices in sorted(cluster_map.items()):
		print(f"  Cluster {cluster_id}: {len(obs_indices)} observations")


if __name__ == '__main__':
	main()
