"""Analyze intersection and rank differences between lottery tickets from two paths."""

import argparse
import numpy as np
import os
from pathlib import Path


def load_tickets(noise_path, top_k=None):
	"""Load lottery tickets from noise_samples.npy.
	
	Args:
		top_k: If specified, return only top-k tickets. Otherwise return all.
	"""
	noise_file = os.path.join(noise_path, "noise_samples.npy")
	if not os.path.exists(noise_file):
		raise FileNotFoundError(f"No noise_samples.npy found in {noise_path}")
	
	tickets = np.load(noise_file)
	
	if top_k is not None:
		if top_k > len(tickets):
			print(f"Warning: Requested top_k={top_k} but only {len(tickets)} tickets available")
			top_k = len(tickets)
		return tickets[:top_k]
	
	return tickets


def compute_ticket_similarity(ticket_a, ticket_b, threshold=1e-6):
	"""Check if two tickets are similar (within threshold)."""
	return np.allclose(ticket_a, ticket_b, atol=threshold)


def find_ticket_matches(tickets_a, tickets_b, threshold=1e-6):
	"""Find matching tickets between two sets and return mapping.
	
	Returns:
		matches_a_to_b: dict mapping index in A to index in B (or None if not found)
		matches_b_to_a: dict mapping index in B to index in A (or None if not found)
	"""
	matches_a_to_b = {}
	matches_b_to_a = {}
	
	# Find all tickets in B (not just top-k)
	all_tickets_b_path = os.path.dirname(os.path.dirname(tickets_b.__class__.__name__))  # Will be replaced below
	
	for i, ticket_a in enumerate(tickets_a):
		for j, ticket_b in enumerate(tickets_b):
			if compute_ticket_similarity(ticket_a, ticket_b, threshold):
				matches_a_to_b[i] = j
				if j not in matches_b_to_a:
					matches_b_to_a[j] = i
				break  # Found match for this ticket_a
	
	return matches_a_to_b, matches_b_to_a


def find_all_ticket_ranks(top_k_tickets, all_tickets, threshold=1e-6):
	"""Find where each top-k ticket appears in the full set of all tickets.
	
	Returns:
		dict mapping top-k index to rank in all_tickets (None if not found)
	"""
	ranks = {}
	
	for i, ticket in enumerate(top_k_tickets):
		found = False
		for j, all_ticket in enumerate(all_tickets):
			if compute_ticket_similarity(ticket, all_ticket, threshold):
				ranks[i] = j
				found = True
				break
		if not found:
			ranks[i] = None
	
	return ranks


def analyze_intersection(path_a, path_b, top_k=100, threshold=1e-6):
	"""Analyze intersection and rank differences between two lottery ticket paths."""
	print(f"Loading top-{top_k} tickets from path A: {path_a}")
	tickets_a_top = load_tickets(path_a, top_k)
	print(f"  Loaded {len(tickets_a_top)} tickets")
	
	print(f"Loading top-{top_k} tickets from path B: {path_b}")
	tickets_b_top = load_tickets(path_b, top_k)
	print(f"  Loaded {len(tickets_b_top)} tickets")
	
	print(f"\nLoading all tickets from both paths for rank analysis...")
	tickets_a_all = load_tickets(path_a)
	tickets_b_all = load_tickets(path_b)
	print(f"  Path A: {len(tickets_a_all)} total tickets")
	print(f"  Path B: {len(tickets_b_all)} total tickets")
	
	print(f"\nFinding intersection in top-{top_k} (threshold={threshold})...")
	matches_a_to_b = {}
	matches_b_to_a = {}
	
	for i, ticket_a in enumerate(tickets_a_top):
		for j, ticket_b in enumerate(tickets_b_top):
			if compute_ticket_similarity(ticket_a, ticket_b, threshold):
				matches_a_to_b[i] = j
				if j not in matches_b_to_a:
					matches_b_to_a[j] = i
				break
	
	n_matches = len(matches_a_to_b)
	print(f"\n{'='*80}")
	print(f"INTERSECTION ANALYSIS (TOP-{top_k})")
	print(f"{'='*80}")
	print(f"Number of matching tickets: {n_matches} / {top_k}")
	print(f"Intersection rate: {n_matches / top_k * 100:.2f}%")
	
	# Find ranks of top-k tickets from A in all of B
	print(f"\n{'='*80}")
	print(f"RANK DIFFERENCES: Top-{top_k} of A in all of B")
	print(f"{'='*80}")
	
	ranks_a_in_b = find_all_ticket_ranks(tickets_a_top, tickets_b_all, threshold)
	
	rank_diffs_a = []
	not_found_a = []
	for idx_a in range(len(tickets_a_top)):
		rank_b = ranks_a_in_b[idx_a]
		if rank_b is not None:
			rank_diff = rank_b - idx_a
			rank_diffs_a.append(rank_diff)
			if abs(rank_diff) > 10 or idx_a < 10:
				print(f"  Ticket A[{idx_a:3d}] -> B[{rank_b:4d}]  (diff: {rank_diff:+5d})")
		else:
			not_found_a.append(idx_a)
			if idx_a < 10:
				print(f"  Ticket A[{idx_a:3d}] -> NOT FOUND in B")
	
	if rank_diffs_a:
		rank_diffs_a = np.array(rank_diffs_a)
		print(f"\nRank difference statistics (A -> B):")
		print(f"  Found: {len(rank_diffs_a)} / {top_k}")
		print(f"  Mean: {rank_diffs_a.mean():+.2f}")
		print(f"  Std:  {rank_diffs_a.std():.2f}")
		print(f"  Min:  {rank_diffs_a.min():+d}")
		print(f"  Max:  {rank_diffs_a.max():+d}")
	if not_found_a:
		print(f"  Not found in B: {len(not_found_a)} tickets")
	
	# Find ranks of top-k tickets from B in all of A
	print(f"\n{'='*80}")
	print(f"RANK DIFFERENCES: Top-{top_k} of B in all of A")
	print(f"{'='*80}")
	
	ranks_b_in_a = find_all_ticket_ranks(tickets_b_top, tickets_a_all, threshold)
	
	rank_diffs_b = []
	not_found_b = []
	for idx_b in range(len(tickets_b_top)):
		rank_a = ranks_b_in_a[idx_b]
		if rank_a is not None:
			rank_diff = rank_a - idx_b
			rank_diffs_b.append(rank_diff)
			if abs(rank_diff) > 10 or idx_b < 10:
				print(f"  Ticket B[{idx_b:3d}] -> A[{rank_a:4d}]  (diff: {rank_diff:+5d})")
		else:
			not_found_b.append(idx_b)
			if idx_b < 10:
				print(f"  Ticket B[{idx_b:3d}] -> NOT FOUND in A")
	
	if rank_diffs_b:
		rank_diffs_b = np.array(rank_diffs_b)
		print(f"\nRank difference statistics (B -> A):")
		print(f"  Found: {len(rank_diffs_b)} / {top_k}")
		print(f"  Mean: {rank_diffs_b.mean():+.2f}")
		print(f"  Std:  {rank_diffs_b.std():.2f}")
		print(f"  Min:  {rank_diffs_b.min():+d}")
		print(f"  Max:  {rank_diffs_b.max():+d}")
	if not_found_b:
		print(f"  Not found in A: {len(not_found_b)} tickets")
	
	# Summary
	print(f"\n{'='*80}")
	print(f"SUMMARY")
	print(f"{'='*80}")
	print(f"Intersection in top-{top_k}: {n_matches} ({n_matches / top_k * 100:.2f}%)")
	if len(rank_diffs_a) > 0:
		print(f"Top-{top_k} of A found in B: {len(rank_diffs_a)} ({len(rank_diffs_a) / top_k * 100:.2f}%)")
		print(f"  Average rank in B: {rank_diffs_a.mean():+.2f}")
	if len(rank_diffs_b) > 0:
		print(f"Top-{top_k} of B found in A: {len(rank_diffs_b)} ({len(rank_diffs_b) / top_k * 100:.2f}%)")
		print(f"  Average rank in A: {rank_diffs_b.mean():+.2f}")


def main():
	parser = argparse.ArgumentParser(description="Analyze intersection between lottery tickets from two paths")
	parser.add_argument("--path_a", type=str, required=True, help="Path to first noise samples directory")
	parser.add_argument("--path_b", type=str, required=True, help="Path to second noise samples directory")
	parser.add_argument("--top_k", type=int, default=100, help="Number of top tickets to compare")
	parser.add_argument("--threshold", type=float, default=1e-6, help="Similarity threshold for matching tickets")
	
	args = parser.parse_args()
	
	analyze_intersection(args.path_a, args.path_b, args.top_k, args.threshold)


if __name__ == "__main__":
	main()
