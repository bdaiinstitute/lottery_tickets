#!/usr/bin/env python3
"""
Script to print per-task success rates and episode counts from eval_info.json
Can also compare two eval files when two paths are provided.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_data(eval_json_path: str) -> Tuple[Dict, Path]:
    """
    Load evaluation data from JSON file

    Args:
        eval_json_path: Path to the eval_info.json file

    Returns:
        Tuple of (data dict, Path object)
    """
    eval_path = Path(eval_json_path)

    if not eval_path.exists():
        raise FileNotFoundError(f"File not found: {eval_json_path}")

    with open(eval_path, "r") as f:
        data = json.load(f)

    return data, eval_path


def extract_task_results(data: Dict) -> List[Dict]:
    """
    Extract per-task results from eval data

    Args:
        data: Loaded JSON data

    Returns:
        List of task result dictionaries with computed metrics
    """
    if "all_results" not in data or len(data["all_results"]) == 0:
        return []

    result = data["all_results"][0]
    if "per_task" not in result:
        return []

    task_results = []
    for task in result["per_task"]:
        task_id = task.get("task_id", "unknown")
        task_group = task.get("task_group", "unknown")

        if "metrics" in task and "sum_rewards" in task["metrics"]:
            rewards = task["metrics"]["sum_rewards"]
            num_episodes = len(rewards)
            success_count = sum(1 for r in rewards if r > 0)
            success_rate = success_count / num_episodes if num_episodes > 0 else 0

            task_results.append(
                {
                    "task_id": task_id,
                    "task_group": task_group,
                    "num_episodes": num_episodes,
                    "success_count": success_count,
                    "success_rate": success_rate,
                }
            )

    return task_results


def print_eval_results(eval_json_path: str):
    """
    Parse and print evaluation results from eval_info.json

    Args:
        eval_json_path: Path to the eval_info.json file
    """
    try:
        data, eval_path = load_eval_data(eval_json_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"\n{'='*70}")
    print(f"Evaluation Results: {eval_path.name}")
    print(f"{'='*70}\n")

    task_results = extract_task_results(data)

    if not task_results:
        print("Error: No valid task results found in JSON")
        return

    print(f"Number of tasks: {len(task_results)}\n")

    total_episodes = 0
    total_successes = 0

    # Print per-task results
    for task in task_results:
        task_id = task["task_id"]
        task_group = task["task_group"]
        num_episodes = task["num_episodes"]
        success_count = task["success_count"]
        success_rate = task["success_rate"]

        total_episodes += num_episodes
        total_successes += success_count

        # Highlight low-performing tasks
        warning = " ⚠️ LOW" if success_rate < 0.5 else ""

        print(f"Task {task_id} ({task_group}):")
        print(f"  Episodes: {num_episodes}")
        print(
            f"  Success Rate: {success_rate:.2%} ({success_count}/{num_episodes}){warning}"
        )
        print()

    # Print summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tasks: {len(task_results)}")
    print(
        f"Episodes per Task: {total_episodes // len(task_results) if len(task_results) > 0 else 0}"
    )
    print(f"Total Episodes: {total_episodes}")
    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0
    print(
        f"Overall Success Rate: {overall_rate:.2%} ({total_successes}/{total_episodes})"
    )
    print(f"{'='*70}\n")


def compare_eval_results(eval_json_path1: str, eval_json_path2: str):
    """
    Compare two evaluation results files

    Args:
        eval_json_path1: Path to the first eval_info.json file
        eval_json_path2: Path to the second eval_info.json file
    """
    try:
        data1, path1 = load_eval_data(eval_json_path1)
        data2, path2 = load_eval_data(eval_json_path2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    task_results1 = extract_task_results(data1)
    task_results2 = extract_task_results(data2)

    if not task_results1 or not task_results2:
        print("Error: One or both files have no valid task results")
        return

    print(f"\n{'='*80}")
    print("COMPARISON: Eval Results")
    print(f"{'='*80}")
    print(f"File 1: {path1.parent.name}/{path1.name}")
    print(f"File 2: {path2.parent.name}/{path2.name}")
    print(f"{'='*80}\n")

    # Create task lookup for file 2
    task2_lookup = {task["task_id"]: task for task in task_results2}

    # Compare per-task results
    print(f"{'Task':>6} {'File1 SR':>12} {'File2 SR':>12} {'Difference':>12}")
    print(f"{'-'*60}")

    total_eps1 = 0
    total_succ1 = 0
    total_eps2 = 0
    total_succ2 = 0

    better_count = 0
    worse_count = 0
    same_count = 0

    for task1 in task_results1:
        task_id = task1["task_id"]
        sr1 = task1["success_rate"]
        eps1 = task1["num_episodes"]
        succ1 = task1["success_count"]

        total_eps1 += eps1
        total_succ1 += succ1

        if task_id in task2_lookup:
            task2 = task2_lookup[task_id]
            sr2 = task2["success_rate"]
            eps2 = task2["num_episodes"]
            succ2 = task2["success_count"]

            total_eps2 += eps2
            total_succ2 += succ2

            diff = sr2 - sr1

            # Track statistics
            if abs(diff) < 0.001:  # Within 0.1%
                same_count += 1
            elif diff > 0:
                better_count += 1
            else:
                worse_count += 1

            # Format difference
            diff_str = f"{diff:+.2%}"

            print(f"{task_id:>6} {sr1:>11.2%} {sr2:>11.2%} {diff_str:>12}")
        else:
            print(f"{task_id:>6} {sr1:>11.2%} {'N/A':>12} {'N/A':>12}")

    # Check for tasks in file 2 not in file 1
    for task2 in task_results2:
        if task2["task_id"] not in [t["task_id"] for t in task_results1]:
            sr2 = task2["success_rate"]
            print(f"{task2['task_id']:>6} {'N/A':>12} {sr2:>11.2%} {'N/A':>12}")

    print(f"{'-'*60}")

    # Overall comparison
    overall_sr1 = total_succ1 / total_eps1 if total_eps1 > 0 else 0
    overall_sr2 = total_succ2 / total_eps2 if total_eps2 > 0 else 0
    overall_diff = overall_sr2 - overall_sr1

    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")
    print(f"File 1 Overall: {overall_sr1:.2%} ({total_succ1}/{total_eps1})")
    print(f"File 2 Overall: {overall_sr2:.2%} ({total_succ2}/{total_eps2})")
    print(f"Difference:     {overall_diff:+.2%}")
    print()
    print(f"Per-Task Summary:")
    print(f"  Better:  {better_count} tasks")
    print(f"  Worse:   {worse_count} tasks")
    print(f"  Same:    {same_count} tasks")

    if overall_diff > 0:
        print(f"\n✓ File 2 performs BETTER overall by {overall_diff:.2%}")
    elif overall_diff < 0:
        print(f"\n✗ File 2 performs WORSE overall by {overall_diff:.2%}")
    else:
        print(f"\n= Files have IDENTICAL overall performance")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Print per-task success rates from eval_info.json or compare two files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print results from a single file
  python print_eval_results.py outputs/libero_spatial_tickets/ticket_results/eval_info.json
  
  # Compare two files
  python print_eval_results.py outputs/libero_spatial_tickets/ticket_results/eval_info.json \\
                               outputs/libero_object_tickets/a2dc9343ba1941199411e7635314a71a/eval_info.json
        """,
    )
    parser.add_argument(
        "eval_json_path", type=str, help="Path to the first eval_info.json file"
    )
    parser.add_argument(
        "eval_json_path2",
        type=str,
        nargs="?",
        default=None,
        help="(Optional) Path to the second eval_info.json file for comparison",
    )

    args = parser.parse_args()

    if args.eval_json_path2:
        # Compare two files
        compare_eval_results(args.eval_json_path, args.eval_json_path2)
    else:
        # Print single file
        print_eval_results(args.eval_json_path)


if __name__ == "__main__":
    main()
