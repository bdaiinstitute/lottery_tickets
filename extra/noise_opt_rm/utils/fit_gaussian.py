"""Fit a Gaussian distribution to lottery ticket data."""

import argparse
import os
import pickle
import numpy as np
from pathlib import Path


def p_args():
	p = argparse.ArgumentParser()
	p.add_argument("--noise_path", type=str, required=True, help="Path to noise_samples.npy file")
	p.add_argument("--top_k", type=int, default=50, help="Number of top tickets to use for fitting")
	p.add_argument("--out", type=str, default=None, help="Output path for the fitted model (default: same dir as noise_path)")
	return p.parse_args()


def fit_zero_mean_gaussian(noise_samples, top_k):
	"""Fit a zero-mean Gaussian to the top-k noise samples.
	
	Args:
		noise_samples: (N, D) array of noise vectors
		top_k: Number of top samples to use
	
	Returns:
		cov: (D, D) covariance matrix
	"""
	# Select top-k samples
	top_samples = noise_samples[:top_k]
	
	# Compute covariance matrix (assuming zero mean)
	cov = np.cov(top_samples.T)
	
	return cov


def main():
	args = p_args()
	
	# Load noise samples
	if not os.path.exists(args.noise_path):
		raise FileNotFoundError(f"Noise file not found: {args.noise_path}")
	
	noise_samples = np.load(args.noise_path)
	print(f"Loaded {len(noise_samples)} noise samples from {args.noise_path}")
	print(f"Noise shape: {noise_samples.shape}")
	
	if args.top_k > len(noise_samples):
		print(f"Warning: top_k={args.top_k} exceeds available samples ({len(noise_samples)}). Using all samples.")
		args.top_k = len(noise_samples)
	
	# Fit zero-mean Gaussian
	print(f"Fitting zero-mean Gaussian using top {args.top_k} samples...")
	cov = fit_zero_mean_gaussian(noise_samples, args.top_k)
	
	# Prepare model
	model = {
		"mean": np.zeros(noise_samples.shape[1]),
		"cov": cov,
		"top_k": args.top_k,
		"noise_path": args.noise_path
	}
	
	# Determine output path
	if args.out is None:
		noise_dir = os.path.dirname(args.noise_path)
		out_path = os.path.join(noise_dir, f"gaussian_model_top{args.top_k}.pkl")
	else:
		out_path = args.out
	
	# Save model
	with open(out_path, "wb") as f:
		pickle.dump(model, f)
	
	print(f"Saved Gaussian model to: {out_path}")
	print(f"  Mean: {model['mean'].shape} (all zeros)")
	print(f"  Covariance: {model['cov'].shape}")
	print(f"  Fitted on top-{args.top_k} samples")


if __name__ == "__main__":
	main()
