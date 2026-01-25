"""Visualize initial observations using t-SNE dimensionality reduction."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE


def load_success_data(result_dir):
	"""Load success data and ranking from result directory."""
	# Load ranking to get top noise indices
	ranking_path = os.path.join(result_dir, 'ranking.txt')
	top_noise_indices = []
	with open(ranking_path, 'r') as f:
		for line in f:
			parts = line.strip().split('\t')
			for part in parts:
				if part.startswith('idx='):
					idx = int(part.split('=')[1])
					top_noise_indices.append(idx)
					break
	
	# Load successes array (sorted by ranking)
	successes = np.load(os.path.join(result_dir, 'successes.npy'), allow_pickle=True)
	
	# Load summary to get original indices mapping
	with open(os.path.join(result_dir, 'summary.json'), 'r') as f:
		summary = json.load(f)
	
	return successes, top_noise_indices, summary


def visualize_initial_obs(obs_path, output_path=None, perplexities=None, n_iter=5000, random_state=42, 
                          top_k_noises=8, result_dir=None, cluster_labels_path=None):
	"""
	Visualize high-dimensional initial observations in 2D using t-SNE.
	
	Args:
		obs_path: Path to initial_obs.npy file with shape (n_envs, obs_dim)
		output_path: Base path to save the visualizations (default: same dir as obs_path)
		perplexities: List of perplexity values to try (default: [2, 5, 10, 30, 50, 100])
		n_iter: Number of t-SNE iterations (default: 1000)
		random_state: Random seed for reproducibility (default: 42)
		top_k_noises: Number of top noise samples to visualize (default: 8)
		result_dir: Directory containing success data (default: same as obs_path)
		cluster_labels_path: Path to cluster_labels.npy for cluster-based coloring (default: None)
	"""
	# Load initial observations
	initial_obs = np.load(obs_path)
	print(f"Loaded initial observations with shape: {initial_obs.shape}")
	n_envs, obs_dim = initial_obs.shape
	
	# Load cluster labels if provided
	cluster_labels = None
	n_clusters = None
	if cluster_labels_path is not None:
		if os.path.exists(cluster_labels_path):
			cluster_labels = np.load(cluster_labels_path)
			n_clusters = len(np.unique(cluster_labels))
			print(f"Loaded cluster labels with {n_clusters} clusters")
		else:
			print(f"Warning: Cluster labels path not found: {cluster_labels_path}")
	
	# Determine result directory
	if result_dir is None:
		result_dir = os.path.dirname(obs_path)
	
	# Load success data
	print("Loading success data...")
	successes, top_noise_indices, summary = load_success_data(result_dir)
	print(f"Loaded success data for {len(top_noise_indices)} noise samples")
	
	# Select top-k noise indices
	top_k_indices = top_noise_indices[:top_k_noises]
	print(f"Visualizing top {top_k_noises} noise samples: {top_k_indices}")
	
	# Set default perplexities
	if perplexities is None:
		perplexities = [2, 5, 10, 30, 50, 100]
	
	# Filter perplexities to valid range (must be less than n_samples)
	max_perplexity = n_envs - 1
	valid_perplexities = [p for p in perplexities if p < max_perplexity]
	if len(valid_perplexities) < len(perplexities):
		print(f"Note: Limiting perplexities to max={max_perplexity} based on n_envs={n_envs}")
	
	if not valid_perplexities:
		valid_perplexities = [min(30, max_perplexity)]
	
	print(f"Will generate visualizations for perplexities: {valid_perplexities}")
	
	# Determine base output path and create t-SNE subdirectory
	if output_path is None:
		base_output_dir = os.path.dirname(obs_path)
	else:
		base_output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else os.path.dirname(obs_path)
	
	# Create tsne_vis subdirectory
	tsne_dir = os.path.join(base_output_dir, 'tsne_vis')
	os.makedirs(tsne_dir, exist_ok=True)
	print(f"Saving visualizations to: {tsne_dir}")
	
	# Run t-SNE for each perplexity
	for perplexity in valid_perplexities:
		print(f"\n{'='*60}")
		print(f"Running t-SNE with perplexity={perplexity}")
		print(f"{'='*60}")
		
		# Apply t-SNE
		tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state, verbose=1)
		obs_2d = tsne.fit_transform(initial_obs)
		
		# Save the 2D coordinates
		coords_path = os.path.join(tsne_dir, f'initial_obs_tsne_perp{perplexity}_coords.npy')
		np.save(coords_path, obs_2d)
		print(f"Saved 2D coordinates to: {coords_path}")
		
		# Create grid visualization (success/failure coloring)
		n_cols = 4
		n_rows = (top_k_noises + n_cols - 1) // n_cols
		
		fig = plt.figure(figsize=(20, 5 * n_rows))
		gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)
		
		for idx, noise_rank in enumerate(top_k_indices):
			row = idx // n_cols
			col = idx % n_cols
			ax = fig.add_subplot(gs[row, col])
			
			# Get success flags for this noise sample (sorted order in successes array)
			success_flags = successes[idx].astype(bool)  # idx corresponds to rank
			
			# Get success rate
			success_rate = summary['success_rates_sorted'][idx]
			original_idx = noise_rank
			
			# Create color array: success=green (1), failure=red (0)
			colors = np.where(success_flags, 'green', 'red')
			
			# Plot points
			scatter = ax.scatter(obs_2d[:, 0], obs_2d[:, 1], c=colors, alpha=0.6, s=30, edgecolors='none')
			
			ax.set_xlabel('t-SNE Dim 1', fontsize=10)
			ax.set_ylabel('t-SNE Dim 2', fontsize=10)
			ax.set_title(f'Rank {idx} (Noise #{original_idx})\nSuccess Rate: {success_rate:.2%}', 
			            fontsize=11, fontweight='bold')
			ax.grid(True, alpha=0.3, linestyle='--')
			
			# Add success/failure counts
			n_success = success_flags.sum()
			n_failure = (~success_flags).sum()
			ax.text(0.02, 0.98, f'✓ {n_success}  ✗ {n_failure}', 
			       transform=ax.transAxes, fontsize=9, verticalalignment='top',
			       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
		
		# Overall title
		fig.suptitle(f't-SNE Visualization (perplexity={perplexity}) - Top {top_k_noises} Noise Samples\n' + 
		            f'Green = Success, Red = Failure | {n_envs} environments, {obs_dim}D → 2D',
		            fontsize=16, fontweight='bold', y=0.995)
		
		plt.tight_layout()
		
		# Save grid figure
		output_file = os.path.join(tsne_dir, f'initial_obs_tsne_perp{perplexity}_grid.png')
		plt.savefig(output_file, dpi=150, bbox_inches='tight')
		print(f"Saved grid visualization to: {output_file}")
		
		plt.close()
		
		# Print some statistics
		print(f"2D Coordinates Statistics:")
		print(f"  Dim 1 range: [{obs_2d[:, 0].min():.2f}, {obs_2d[:, 0].max():.2f}]")
		print(f"  Dim 2 range: [{obs_2d[:, 1].min():.2f}, {obs_2d[:, 1].max():.2f}]")
		print(f"  Mean distance between points: {np.mean(np.linalg.norm(obs_2d[1:] - obs_2d[:-1], axis=1)):.2f}")
		
		# If cluster labels are provided, create additional cluster-colored visualization
		if cluster_labels is not None:
			print(f"\nCreating cluster-colored visualization...")
			
			# Create a colormap for clusters
			cmap = plt.cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
			cluster_colors = [cmap(i / n_clusters) for i in range(n_clusters)]
			
			# Create grid for cluster visualization
			fig_cluster = plt.figure(figsize=(20, 5 * n_rows))
			gs_cluster = GridSpec(n_rows, n_cols, figure=fig_cluster, hspace=0.3, wspace=0.25)
			
			for idx, noise_rank in enumerate(top_k_indices):
				row = idx // n_cols
				col = idx % n_cols
				ax = fig_cluster.add_subplot(gs_cluster[row, col])
				
				# Get success rate
				success_rate = summary['success_rates_sorted'][idx]
				original_idx = noise_rank
				
				# Color by cluster
				point_colors = [cluster_colors[label] for label in cluster_labels]
				
				# Calculate cluster success rates first for labels
				success_flags = successes[idx].astype(bool)
				cluster_success_rates = {}
				for c in range(n_clusters):
					mask = cluster_labels == c
					if mask.sum() > 0:
						cluster_success_rates[c] = success_flags[mask].mean()
				
				# Plot points with success rate in label
				for c in range(n_clusters):
					mask = cluster_labels == c
					success_rate_c = cluster_success_rates.get(c, 0.0)
					ax.scatter(obs_2d[mask, 0], obs_2d[mask, 1], 
					          c=[cluster_colors[c]], alpha=0.6, s=30, 
					          edgecolors='black', linewidths=0.5, 
					          label=f'C{c}: {success_rate_c:.1%}')
				
				ax.set_xlabel('t-SNE Dim 1', fontsize=10)
				ax.set_ylabel('t-SNE Dim 2', fontsize=10)
				ax.set_title(f'Rank {idx} (Noise #{original_idx})\nSuccess Rate: {success_rate:.2%}', 
				            fontsize=11, fontweight='bold')
				ax.grid(True, alpha=0.3, linestyle='--')
				
				# Add legend with cluster colors and success rates (sorted by success rate)
				all_clusters = sorted(cluster_success_rates.items(), key=lambda x: x[1], reverse=True)
				handles, labels = ax.get_legend_handles_labels()
				# Reorder legend by success rate
				sorted_handles_labels = [(h, l) for _, (h, l) in sorted(
					zip(range(n_clusters), zip(handles, labels)), 
					key=lambda x: cluster_success_rates.get(x[0], 0), 
					reverse=True
				)]
				sorted_handles, sorted_labels = zip(*sorted_handles_labels) if sorted_handles_labels else ([], [])
				ax.legend(sorted_handles, sorted_labels, loc='upper right', fontsize=7, ncol=2, framealpha=0.9, 
				         title='Clusters (by success rate)', title_fontsize=7)
			
			# Overall title
			fig_cluster.suptitle(f't-SNE Visualization with K-means Clusters (perplexity={perplexity})\n' + 
			                    f'Top {top_k_noises} Noise Samples | {n_clusters} clusters | {n_envs} environments, {obs_dim}D → 2D',
			                    fontsize=16, fontweight='bold', y=0.995)
			
			plt.tight_layout()
			
			# Save cluster-colored grid figure
			output_file_cluster = os.path.join(tsne_dir, f'initial_obs_tsne_perp{perplexity}_grid_clusters_k{n_clusters}.png')
			plt.savefig(output_file_cluster, dpi=150, bbox_inches='tight')
			print(f"Saved cluster-colored visualization to: {output_file_cluster}")
			
			plt.close()


def main():
	parser = argparse.ArgumentParser(description='Visualize initial observations using t-SNE with success/failure overlay')
	parser.add_argument('obs_path', type=str, help='Path to initial_obs.npy file')
	parser.add_argument('--output', type=str, default=None, help='Output path for visualization (default: same dir as input)')
	parser.add_argument('--perplexities', type=int, nargs='+', default=None, help='t-SNE perplexity values (default: [2, 5, 10, 30, 50, 100])')
	parser.add_argument('--n_iter', type=int, default=1000, help='Number of t-SNE iterations (default: 1000)')
	parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
	parser.add_argument('--top_k', type=int, default=8, help='Number of top noise samples to visualize (default: 8)')
	parser.add_argument('--result_dir', type=str, default=None, help='Directory containing result files (default: same as obs_path)')
	parser.add_argument('--cluster_labels', type=str, default=None, help='Path to cluster_labels.npy for cluster-based coloring (default: None)')
	parser.add_argument('--kmeans', action='store_true', help='Use K-means cluster labels from clustering/ subdirectory')
	
	args = parser.parse_args()
	
	if not os.path.exists(args.obs_path):
		raise FileNotFoundError(f"File not found: {args.obs_path}")
	
	# If --kmeans flag is set, automatically find cluster_labels.npy in clustering/ subdirectory
	cluster_labels_path = args.cluster_labels
	if args.kmeans:
		obs_dir = os.path.dirname(args.obs_path)
		cluster_labels_path = os.path.join(obs_dir, 'clustering', 'cluster_labels.npy')
		if not os.path.exists(cluster_labels_path):
			print(f"Warning: --kmeans flag set but cluster_labels.npy not found at: {cluster_labels_path}")
			cluster_labels_path = None
		else:
			print(f"Using K-means cluster labels from: {cluster_labels_path}")
	
	visualize_initial_obs(
		args.obs_path, 
		args.output, 
		args.perplexities, 
		args.n_iter, 
		args.seed,
		args.top_k,
		args.result_dir,
		cluster_labels_path
	)


if __name__ == '__main__':
	main()
