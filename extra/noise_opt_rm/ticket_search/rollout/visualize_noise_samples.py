"""Visualize noise samples using t-SNE with performance-based coloring."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from sklearn.manifold import TSNE


def visualize_noise_samples(result_dir, output_path=None, perplexities=None, n_iter=5000, random_state=42):
	"""
	Visualize noise samples in 2D using t-SNE, colored by performance.
	
	Args:
		result_dir: Path to lottery ticket results directory containing:
			- noise_samples.npy: (n_noise_samples, noise_dim) array of noise vectors
			- summary.json: contains success_rates_sorted for coloring
		output_path: Base path to save the visualizations (default: same dir as result_dir)
		perplexities: List of perplexity values to try (default: [2, 5, 10, 30, 50, 100])
		n_iter: Number of t-SNE iterations (default: 5000)
		random_state: Random seed for reproducibility (default: 42)
	"""
	print(f"Visualizing noise samples from: {result_dir}")
	
	# Load noise samples
	noise_samples = np.load(os.path.join(result_dir, 'noise_samples.npy'))
	print(f"Loaded noise samples with shape: {noise_samples.shape}")
	n_noise_samples, noise_dim = noise_samples.shape
	
	# Load summary for success rates
	with open(os.path.join(result_dir, 'summary.json'), 'r') as f:
		summary = json.load(f)
	
	# Get success rates (these are already sorted by performance)
	success_rates = np.array(summary['success_rates_sorted'])
	print(f"Success rate range: [{success_rates.min():.2%}, {success_rates.max():.2%}]")
	
	# Set default perplexities
	if perplexities is None:
		perplexities = [2, 5, 10, 30, 50, 100]
	
	# Filter perplexities to valid range (must be less than n_samples)
	max_perplexity = n_noise_samples - 1
	valid_perplexities = [p for p in perplexities if p < max_perplexity]
	if len(valid_perplexities) < len(perplexities):
		print(f"Note: Limiting perplexities to max={max_perplexity} based on n_noise_samples={n_noise_samples}")
	
	if not valid_perplexities:
		valid_perplexities = [min(30, max_perplexity)]
	
	print(f"Will generate visualizations for perplexities: {valid_perplexities}")
	
	# Determine output directory
	if output_path is None:
		output_dir = result_dir
	else:
		output_dir = output_path
	
	# Create noise_tsne_vis subdirectory
	tsne_dir = os.path.join(output_dir, 'noise_tsne_vis')
	os.makedirs(tsne_dir, exist_ok=True)
	print(f"Saving visualizations to: {tsne_dir}")
	
	# Run t-SNE for each perplexity
	for perplexity in valid_perplexities:
		print(f"\n{'='*60}")
		print(f"Running t-SNE with perplexity={perplexity}")
		print(f"{'='*60}")
		
		# Apply t-SNE
		tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state, verbose=1)
		noise_2d = tsne.fit_transform(noise_samples)
		
		# Save the 2D coordinates
		coords_path = os.path.join(tsne_dir, f'noise_tsne_perp{perplexity}_coords.npy')
		np.save(coords_path, noise_2d)
		print(f"Saved 2D coordinates to: {coords_path}")
		
		# Compute additional metrics
		# 1. Magnitude of noise vectors
		noise_magnitudes = np.linalg.norm(noise_samples, axis=1)
		
		# 2. Log probability of samples under standard normal distribution
		# For multivariate standard normal: log p(x) = -0.5 * ||x||^2 - 0.5 * d * log(2π)
		# We'll use negative log probability for better visualization (lower = more probable)
		log_probs = -0.5 * np.sum(noise_samples**2, axis=1) - 0.5 * noise_dim * np.log(2 * np.pi)
		# Convert to probability-like metric (higher = more probable)
		# Use exp of normalized log probs to get relative probabilities
		prob_metric = np.exp(log_probs - log_probs.max())  # Normalize by max for numerical stability
		
		# 3. Cosine similarity with unit vector (all ones normalized)
		unit_vector = np.ones(noise_dim) / np.sqrt(noise_dim)
		cos_similarities = np.dot(noise_samples, unit_vector) / (noise_magnitudes + 1e-8)
		
		# 4. Additional descriptive metrics
		# Mean of noise (directional bias)
		noise_means = noise_samples.mean(axis=1)
		
		# Alignment with top 10% successful noises
		top_10_percent = int(0.1 * n_noise_samples)
		top_noise_indices = np.argsort(success_rates)[::-1][:top_10_percent]
		top_noises_mean = noise_samples[top_noise_indices].mean(axis=0)
		top_noises_mean_normalized = top_noises_mean / (np.linalg.norm(top_noises_mean) + 1e-8)
		alignment_with_top = np.dot(noise_samples, top_noises_mean_normalized) / (noise_magnitudes + 1e-8)
		
		# Create 2x2 subplot figure
		fig = plt.figure(figsize=(16, 12))
		gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)
		
		# Use more drastic alpha range
		alpha_min = 0.05
		alpha_max = 1.0
		normalized_rates = (success_rates - success_rates.min()) / (success_rates.max() - success_rates.min() + 1e-8)
		alphas = alpha_min + (alpha_max - alpha_min) * normalized_rates
		# Plot 1: Success Rate
		ax1 = fig.add_subplot(gs[0, 0])
		scatter1 = ax1.scatter(noise_2d[:, 0], noise_2d[:, 1], 
		                      c=success_rates, cmap='RdYlGn', 
		                      alpha=alphas, s=50, edgecolors='black', linewidths=0.5)
		cbar1 = plt.colorbar(scatter1, ax=ax1)
		cbar1.set_label('Success Rate', fontsize=10)
		cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
		
		ax1.set_xlabel('t-SNE Dim 1', fontsize=10)
		ax1.set_ylabel('t-SNE Dim 2', fontsize=10)
		ax1.set_title(f'Success Rate\n(Mean: {success_rates.mean():.2%}, Std: {success_rates.std():.2%})',
		             fontsize=11, fontweight='bold')
		ax1.grid(True, alpha=0.3, linestyle='--')
		
		# Plot 2: Noise Magnitude
		ax2 = fig.add_subplot(gs[0, 1])
		# Normalize magnitudes for alpha (higher magnitude = more opaque)
		normalized_mags = (noise_magnitudes - noise_magnitudes.min()) / (noise_magnitudes.max() - noise_magnitudes.min() + 1e-8)
		alphas_mag = alpha_min + (alpha_max - alpha_min) * normalized_mags
		
		scatter2 = ax2.scatter(noise_2d[:, 0], noise_2d[:, 1], 
		                      c=noise_magnitudes, cmap='viridis', 
		                      alpha=alphas_mag, s=50, edgecolors='black', linewidths=0.5)
		cbar2 = plt.colorbar(scatter2, ax=ax2)
		cbar2.set_label('||Noise|| (L2 Norm)', fontsize=10)
		
		ax2.set_xlabel('t-SNE Dim 1', fontsize=10)
		ax2.set_ylabel('t-SNE Dim 2', fontsize=10)
		ax2.set_title(f'Noise Magnitude\n(Mean: {noise_magnitudes.mean():.2f}, Std: {noise_magnitudes.std():.2f})',
		             fontsize=11, fontweight='bold')
		ax2.grid(True, alpha=0.3, linestyle='--')
		
		# Plot 3: Probability under Standard Normal
		ax3 = fig.add_subplot(gs[1, 0])
		# Normalize probabilities for alpha (higher probability = more opaque)
		normalized_probs = (prob_metric - prob_metric.min()) / (prob_metric.max() - prob_metric.min() + 1e-8)
		alphas_prob = alpha_min + (alpha_max - alpha_min) * normalized_probs
		
		scatter3 = ax3.scatter(noise_2d[:, 0], noise_2d[:, 1], 
		                      c=prob_metric, cmap='plasma', 
		                      alpha=alphas_prob, s=50, edgecolors='black', linewidths=0.5)
		cbar3 = plt.colorbar(scatter3, ax=ax3)
		cbar3.set_label('Relative Probability\n(Higher = More Typical)', fontsize=10)
		
		ax3.set_xlabel('t-SNE Dim 1', fontsize=10)
		ax3.set_ylabel('t-SNE Dim 2', fontsize=10)
		ax3.set_title(f'Standard Normal Likelihood\n(Log Prob Range: [{log_probs.min():.1f}, {log_probs.max():.1f}])',
		             fontsize=11, fontweight='bold')
		ax3.grid(True, alpha=0.3, linestyle='--')
		
		# Plot 4: Alignment with Top Performers vs Success Rate
		ax4 = fig.add_subplot(gs[1, 1])
		
		scatter4 = ax4.scatter(alignment_with_top, success_rates, 
		                      c=success_rates, cmap='RdYlGn', 
		                      alpha=alphas, s=50, edgecolors='black', linewidths=0.5)
		cbar4 = plt.colorbar(scatter4, ax=ax4)
		cbar4.set_label('Success Rate', fontsize=10)
		cbar4.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
		
		ax4.set_xlabel(f'Alignment with Top {top_10_percent} Noises', fontsize=10)
		ax4.set_ylabel('Success Rate', fontsize=10)
		ax4.set_title(f'Alignment with High Performers vs Success\n(Correlation: {np.corrcoef(alignment_with_top, success_rates)[0,1]:.3f})',
		             fontsize=11, fontweight='bold')
		ax4.grid(True, alpha=0.3, linestyle='--')
		
		# Overall title
		fig.suptitle(f't-SNE Visualization of Noise Samples (perplexity={perplexity}) - {n_noise_samples} samples, {noise_dim}D → 2D',
		            fontsize=14, fontweight='bold', y=1.02)
		
		plt.tight_layout()
		
		# Save figure
		output_file = os.path.join(tsne_dir, f'noise_tsne_perp{perplexity}.png')
		plt.savefig(output_file, dpi=150, bbox_inches='tight')
		print(f"Saved visualization to: {output_file}")
		
		plt.close()
		
		# Print some statistics
		print(f"2D Coordinates Statistics:")
		print(f"  Dim 1 range: [{noise_2d[:, 0].min():.2f}, {noise_2d[:, 0].max():.2f}]")
		print(f"  Dim 2 range: [{noise_2d[:, 1].min():.2f}, {noise_2d[:, 1].max():.2f}]")
		print(f"Noise Magnitude Statistics:")
		print(f"  Range: [{noise_magnitudes.min():.2f}, {noise_magnitudes.max():.2f}]")
		print(f"  Mean: {noise_magnitudes.mean():.2f}, Std: {noise_magnitudes.std():.2f}")
		print(f"Log Probability Statistics:")
		print(f"  Range: [{log_probs.min():.2f}, {log_probs.max():.2f}]")
		print(f"  Mean: {log_probs.mean():.2f}, Std: {log_probs.std():.2f}")
		print(f"Cosine Similarity with Unit Vector Statistics:")
		print(f"  Range: [{cos_similarities.min():.3f}, {cos_similarities.max():.3f}]")
		print(f"  Mean: {cos_similarities.mean():.3f}, Std: {cos_similarities.std():.3f}")


from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def visualize_noise_and_success(results_dir):
	"""
	Analyzes and visualizes the per-dimension statistical differences between
	top and bottom performing noise tickets.
	"""
	# --- Data Loading ---
	try:
		noise_samples = np.load(os.path.join(results_dir, 'noise_samples.npy'))
		success_rates = np.load(os.path.join(results_dir, 'success_rates.npy'))
		print(f"Loaded {len(noise_samples)} noise samples and their success rates.")
	except FileNotFoundError:
		print(f"Error: noise_samples.npy or success_rates.npy not found in '{results_dir}'")
		return

	task_name = os.path.basename(os.path.dirname(results_dir))
	n_samples, noise_dim = noise_samples.shape

	# --- Grouping Tickets: Top 10% vs Bottom 10% ---
	sorted_indices = np.argsort(success_rates)
	top_10_percent_idx = sorted_indices[-int(n_samples * 0.1):]
	bottom_10_percent_idx = sorted_indices[:int(n_samples * 0.1)]

	top_noises = noise_samples[top_10_percent_idx]
	bottom_noises = noise_samples[bottom_10_percent_idx]

	# --- Per-Dimension Statistical Analysis ---
	# Mean and Variance for each of the 588 dimensions
	mean_top = top_noises.mean(axis=0)
	var_top = top_noises.var(axis=0)
	
	mean_bottom = bottom_noises.mean(axis=0)
	var_bottom = bottom_noises.var(axis=0)

	# Get top 10 and bottom 10 individual noise vectors
	top_10_idx = sorted_indices[-10:]
	bottom_10_idx = sorted_indices[:10]
	best_performer_idx = sorted_indices[-1]
	
	top_10_noises = noise_samples[top_10_idx]
	bottom_10_noises = noise_samples[bottom_10_idx]
	best_noise = noise_samples[best_performer_idx]

	# --- Plotting ---
	fig, axes = plt.subplots(2, 2, figsize=(18, 12))
	fig.suptitle(f'Per-Dimension Analysis: Top 10% vs. Bottom 10% Noise Tickets\nTask: {task_name}', fontsize=16, fontweight='bold')
	dims = np.arange(noise_dim)

	# Plot 1: Mean value of each dimension
	ax1 = axes[0, 0]
	ax1.plot(dims, mean_top, color='g', alpha=0.8, label='Top 10% Mean')
	ax1.plot(dims, mean_bottom, color='r', alpha=0.7, linestyle='--', label='Bottom 10% Mean')
	ax1.set_title('Mean of Each Noise Dimension', fontsize=11, fontweight='bold')
	ax1.set_xlabel('Noise Dimension Index')
	ax1.set_ylabel('Mean Value')
	ax1.legend()
	ax1.grid(True, alpha=0.3, linestyle='--')

	# Plot 2: Variance of each dimension
	ax2 = axes[0, 1]
	ax2.plot(dims, var_top, color='g', alpha=0.8, label='Top 10% Variance')
	ax2.plot(dims, var_bottom, color='r', alpha=0.7, linestyle='--', label='Bottom 10% Variance')
	ax2.set_title('Variance of Each Noise Dimension', fontsize=11, fontweight='bold')
	ax2.set_xlabel('Noise Dimension Index')
	ax2.set_ylabel('Variance')
	ax2.legend()
	ax2.grid(True, alpha=0.3, linestyle='--')

	# Plot 3: Scatter of Top 10 and Bottom 10 Individual Samples
	ax3 = axes[1, 0]
	# Plot bottom 10 in red
	for i, noise_vec in enumerate(bottom_10_noises):
		ax3.scatter(dims, noise_vec, color='r', alpha=0.3, s=10)
	# Plot top 10 in green
	for i, noise_vec in enumerate(top_10_noises):
		ax3.scatter(dims, noise_vec, color='g', alpha=0.3, s=10)
	# Highlight the best performer with blue stars
	ax3.scatter(dims, best_noise, color='b', marker='*', s=100, label='Best Performer', alpha=0.9, edgecolors='black', linewidths=0.5)
	
	ax3.set_title('Per-Dimension Values: Top 10 vs. Bottom 10 Samples', fontsize=11, fontweight='bold')
	ax3.set_xlabel('Noise Dimension Index')
	ax3.set_ylabel('Noise Value')
	ax3.legend()
	ax3.grid(True, alpha=0.3, linestyle='--')

	# Plot 4: Empty (removed)
	ax4 = axes[1, 1]
	ax4.axis('off')

	# --- Save and Show Plot ---
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	output_filename = os.path.join(results_dir, f'{task_name}_dimension_analysis.png')
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')
	print(f"Plot saved to {output_filename}")
	plt.show()


def main():
	parser = argparse.ArgumentParser(description='Analyze and visualize noise samples and their success rates.')
	parser.add_argument('results_dir', type=str, help='Path to the directory containing noise_samples.npy and success_rates.npy')
	args = parser.parse_args()

	if not os.path.isdir(args.results_dir):
		print(f"Error: Provided path '{args.results_dir}' is not a valid directory.")
		return

	visualize_noise_and_success(args.results_dir)

if __name__ == '__main__':
	main()
