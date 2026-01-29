#!/usr/bin/env python3
"""
Summarize ticket evaluation results into a table.

Usage:
    python summarize_ticket_results.py <results_dir>

Example:
    python summarize_ticket_results.py /path/to/outputs/libero_spatial_tickets/ticket_results
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_ticket_hash(folder_name):
    """Extract ticket hash from folder name like 'libero_spatial_ticket_<hash>_n50_b10_s1619_<timestamp>'"""
    # Handle special case for original_policy
    if folder_name == 'original_policy' or folder_name.startswith('original_policy'):
        return 'original_policy'
    
    parts = folder_name.split('_')
    # Find the hash part (32 character hex string)
    for part in parts:
        if len(part) == 32 and all(c in '0123456789abcdef' for c in part):
            return part
    return None


def load_eval_info(eval_json_path):
    """Load evaluation info from JSON file."""
    with open(eval_json_path, 'r') as f:
        return json.load(f)


def process_results_directory(results_dir):
    """Process all eval_info.json files in the results directory."""
    results_dir = Path(results_dir)
    
    # Dictionary to store results: ticket_hash -> {task_id: success_rate}
    ticket_results = defaultdict(dict)
    
    # Find all eval_info.json files
    eval_files = list(results_dir.glob("*/eval_info.json"))
    
    if not eval_files:
        print(f"No eval_info.json files found in {results_dir}")
        return None
    
    print(f"Found {len(eval_files)} evaluation result files")
    
    for eval_file in eval_files:
        folder_name = eval_file.parent.name
        ticket_hash = extract_ticket_hash(folder_name)
        
        if not ticket_hash:
            print(f"Warning: Could not extract ticket hash from {folder_name}")
            continue
        
        # Load the evaluation info
        eval_info = load_eval_info(eval_file)
        
        # Extract per-task success rates from all_results
        if 'all_results' in eval_info and len(eval_info['all_results']) > 0:
            result = eval_info['all_results'][0]  # Get first result
            
            if 'per_task' in result:
                for task_info in result['per_task']:
                    task_id = task_info['task_id']
                    metrics = task_info['metrics']
                    
                    # Calculate success rate from successes list
                    if 'successes' in metrics:
                        successes = metrics['successes']
                        success_rate = (sum(successes) / len(successes)) * 100 if successes else 0.0
                        ticket_results[ticket_hash][task_id] = success_rate
    
    return ticket_results


def print_results_table(ticket_results):
    """Print results as a formatted table."""
    if not ticket_results:
        print("No results to display")
        return
    
    # Get all unique task IDs and sort them
    all_task_ids = set()
    for task_dict in ticket_results.values():
        all_task_ids.update(task_dict.keys())
    task_ids = sorted(all_task_ids)
    
    # Sort ticket hashes
    ticket_hashes = sorted(ticket_results.keys())
    
    # Calculate column widths
    hash_width = max(len("Ticket Hash"), max(len(h) for h in ticket_hashes))
    task_width = 10  # Width for each task column
    avg_width = 12   # Width for average column
    
    # Print header
    header = f"{'Ticket Hash':<{hash_width}}"
    for task_id in task_ids:
        header += f" | {'Task ' + str(task_id):>{task_width}}"
    header += f" | {'Avg Success':>{avg_width}}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for ticket_hash in ticket_hashes:
        task_dict = ticket_results[ticket_hash]
        row = f"{ticket_hash:<{hash_width}}"
        
        task_success_rates = []
        for task_id in task_ids:
            success_rate = task_dict.get(task_id, 0.0)
            task_success_rates.append(success_rate)
            row += f" | {success_rate:>{task_width}.1f}%"
        
        # Calculate average
        avg_success = np.mean(task_success_rates)
        row += f" | {avg_success:>{avg_width}.1f}%"
        print(row)
    
    # Print summary statistics
    print("\n" + "=" * len(header))
    print("Summary Statistics:")
    
    # Average per task across all tickets
    print(f"\n{'Task ID':<15} {'Avg Success Rate':>20}")
    print("-" * 40)
    for task_id in task_ids:
        task_rates = [ticket_results[h].get(task_id, 0.0) for h in ticket_hashes]
        avg_rate = np.mean(task_rates)
        std_rate = np.std(task_rates)
        print(f"{'Task ' + str(task_id):<15} {avg_rate:>15.1f}% ± {std_rate:.1f}%")
    
    # Overall average across all tickets and tasks
    all_rates = []
    for task_dict in ticket_results.values():
        all_rates.extend(task_dict.values())
    overall_avg = np.mean(all_rates)
    overall_std = np.std(all_rates)
    print(f"\n{'Overall Average':<15} {overall_avg:>15.1f}% ± {overall_std:.1f}%")


def main():
    if len(sys.argv) != 2:
        print("Usage: python summarize_ticket_results.py <results_dir>")
        print("\nExample:")
        print("  python summarize_ticket_results.py /path/to/outputs/libero_spatial_tickets/ticket_results")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not Path(results_dir).exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)
    
    print(f"Processing results from: {results_dir}\n")
    
    ticket_results = process_results_directory(results_dir)
    
    if ticket_results:
        print_results_table(ticket_results)
    else:
        print("No valid results found")
        sys.exit(1)


if __name__ == "__main__":
    main()
