# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def compute_means_and_success(rewards: list[float], threshold: float) -> tuple[float, ...]:
    """
    Compute first/second half mean reward, standard errors, and success rates.
    
    Args:
        rewards: List of per-episode rewards.
        threshold: Threshold eeward above which to consider an episode successful.

    Returns:
        tuple of
        - first half mean reward
        - second half mean eeward
        - first half standard error
        - second half standard error
        - first half success rate
        - second half success rate
    """
    n = len(rewards)
    split = n // 2
    first = rewards[:split]
    second = rewards[split:]

    first_mean = first.mean()
    second_mean = second.mean()
    first_se = first.std(ddof=1) / np.sqrt(len(first))
    second_se = second.std(ddof=1) / np.sqrt(len(second))

    first_success = (first > threshold).mean()
    second_success = (second > threshold).mean()

    return first_mean, second_mean, first_se, second_se, first_success, second_success


def main(root_dir: Path, out_avg_path: Path, out_success_path: Path, threshold: float = 100.0) -> None:
    """
    Main function for visualizing regression to mean.
    
    Args:
        root_dir: Path to directory containing the model to evaluate.
        out_avg: Output image path for average reward.
        out_success: Output image path for success rate.
        threshold: Episode reward threshold to count as success.
    """
    # Lists for lottery tickets (average rewards)
    x_mean, y_mean, x_se, y_se, ticket_names = [], [], [], [], []

    # Lists for lottery tickets (success rates)
    x_succ, y_succ = [], []

    # Original policy
    orig_name = None
    orig_x, orig_y, orig_x_se, orig_y_se = [], [], [], []
    orig_x_succ, orig_y_succ = [], []
    orig_avg = None
    orig_succ_avg = None

    overall_rewards = {}  # name -> mean(all episodes)
    overall_success = {}  # name -> mean(success rate all episodes)

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        reward_file = subdir / "total_reward_list.npy"
        if not reward_file.exists():
            continue

        rewards = np.load(reward_file)
        n = len(rewards)
        if n < 2:
            continue

        # Compute means, SE, and success rates
        first_mean, second_mean, first_se, second_se, first_succ, second_succ = \
            compute_means_and_success(rewards, threshold)

        avg_reward = rewards.mean()
        success_rate = (rewards > threshold).mean()
        overall_rewards[subdir.name] = avg_reward
        overall_success[subdir.name] = success_rate

        if subdir.name == "original_policy":
            orig_name = subdir.name
            orig_avg = avg_reward
            orig_succ_avg = success_rate

            orig_x.append(first_mean)
            orig_y.append(second_mean)
            orig_x_se.append(first_se)
            orig_y_se.append(second_se)

            orig_x_succ.append(first_succ)
            orig_y_succ.append(second_succ)
        else:
            ticket_names.append(subdir.name)

            x_mean.append(first_mean)
            y_mean.append(second_mean)
            x_se.append(first_se)
            y_se.append(second_se)

            x_succ.append(first_succ)
            y_succ.append(second_succ)

    # Convert to numpy
    x_mean = np.array(x_mean)
    y_mean = np.array(y_mean)
    x_se = np.array(x_se)
    y_se = np.array(y_se)

    x_succ = np.array(x_succ)
    y_succ = np.array(y_succ)

    # Linear regression (exclude original_policy)
    coeffs = np.polyfit(x_mean, y_mean, deg=1)
    a, b = coeffs
    y_pred = a * x_mean + b

    ss_res = np.sum((y_mean - y_pred) ** 2)
    ss_tot = np.sum((y_mean - y_mean.mean()) ** 2)
    r2_reward = 1.0 - ss_res / ss_tot

    # Linear regression for success rate
    if len(x_succ) > 1:
        coeffs_succ = np.polyfit(x_succ, y_succ, deg=1)
        a_s, b_s = coeffs_succ
        y_pred_s = a_s * x_succ + b_s
        ss_res_s = np.sum((y_succ - y_pred_s) ** 2)
        ss_tot_s = np.sum((y_succ - y_succ.mean()) ** 2)
        r2_succ = 1.0 - ss_res_s / ss_tot_s
    else:
        a_s = b_s = r2_succ = np.nan

    # ---- Average reward plot ----
    plt.figure(figsize=(6, 6))
    plt.errorbar(x_mean, y_mean, xerr=x_se, yerr=y_se, fmt='o', alpha=0.7, capsize=2, color='blue', label='Lottery tickets')
    if orig_x:
        plt.errorbar(orig_x, orig_y, xerr=orig_x_se, yerr=orig_y_se, fmt='o', alpha=0.7, capsize=2, color='red', label='Original policy')

    x_line = np.linspace(x_mean.min(), x_mean.max(), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, linestyle='-', color='green', label=f'Best fit (tickets)')

    plt.xlabel(f"Average reward (first 50% episodes, total {n} episodes)")
    plt.ylabel(f"Average reward (second 50% episodes, total {n} episodes)")
    plt.title("Lottery Tickets vs Original Policy (Average Reward)")
    plt.text(0.05, 0.95, f"$R^2 = {r2_reward:.3f}$", transform=plt.gca().transAxes, verticalalignment='top', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_avg_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_avg_path, dpi=200)
    plt.close()

    # ---- Success rate plot ----
    plt.figure(figsize=(6, 6))
    plt.scatter(x_succ, y_succ, color='blue', alpha=0.7, label='Lottery tickets')
    if orig_x_succ:
        plt.scatter(orig_x_succ, orig_y_succ, color='red', alpha=0.7, label='Original policy')

    if len(x_succ) > 1:
        x_line_s = np.linspace(0, 1, 100)
        y_line_s = a_s * x_line_s + b_s
        plt.plot(x_line_s, y_line_s, linestyle='-', color='green', label='Best fit (tickets)')
        plt.text(0.05, 0.95, f"$R^2 = {r2_succ:.3f}$", transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10)

    plt.xlabel(f"Success rate (first 50% episodes, threshold {threshold})")
    plt.ylabel(f"Success rate (second 50% episodes, threshold {threshold})")
    plt.title("Lottery Tickets vs Original Policy (Success Rate)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_success_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_success_path, dpi=200)
    plt.close()

    # Print regression info for reward plot
    print(f"Best-fit line (tickets only, reward): y = {a:.4f} x + {b:.4f}")
    print(f"R^2 (tickets only, reward) = {r2_reward:.4f}\n")
    print(f"Best-fit line (tickets only, success rate): y = {a_s:.4f} x + {b_s:.4f}")
    print(f"R^2 (tickets only, success rate) = {r2_succ:.4f}\n")

    # Print tickets ranked by overall average reward
    print("Tickets ranked by overall average reward:")
    if orig_name:
        print(f"{orig_name}: {orig_avg:.4f}")
    sorted_tickets = sorted(
        ((name, overall_rewards[name]) for name in ticket_names),
        key=lambda x: x[1], reverse=True
    )
    for name, avg in sorted_tickets:
        print(f"{name}: {avg:.4f}")

    # Print tickets ranked by overall success rate
    print("\nTickets ranked by overall task success rate:")
    if orig_name:
        print(f"{orig_name}: {orig_succ_avg:.4f}")
    sorted_tickets_succ = sorted(
        ((name, overall_success[name]) for name in ticket_names),
        key=lambda x: x[1], reverse=True
    )
    for name, succ in sorted_tickets_succ:
        print(f"{name}: {succ:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True, help="Path to directory containing model to evaluate")
    parser.add_argument("--out_avg", type=Path, default=Path("lottery_ticket_avg_reward.png"), help="Output image path for average reward")
    parser.add_argument("--out_success", type=Path, default=Path("lottery_ticket_success_rate.png"), help="Output image path for success rate")
    parser.add_argument("--threshold", type=float, default=100.0, help="Episode reward threshold to count as success")
    args = parser.parse_args()

    main(args.root_dir, args.out_avg, args.out_success, threshold=args.threshold)
