import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for epsilon-tickets")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per epsilon value (default: 100)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Step size for ticket epsilon, ranging from 0 to 1 (default: 0.05)",
    )
    parser.add_argument(
        "--ticket-path",
        type=str,
        default="./envs100_samples5000_seed999_ddim8_20251130_221846_ddim8",
        help="Path to a directory containing saved tickets",
    )

    args = parser.parse_args()
    num_epsilons = int(1.0 / args.step_size) + 1
    num_tickets = np.load(Path(args.ticket_path) / "noise_samples.npy").shape[0]
    epsilons = np.linspace(0, 1, num_epsilons)

    base_command = [
        "python",
        "opt_noise.py",
        "--eval",
        f"{args.ticket_path}",
        "--eval_idx",
        "{ticket_idx}" "--task_name",
        "can",
        "--n_evals_per_seed",
        f"{args.episodes}",
        "--n_seeds",
        "50",
        "--seed",
        "1619",
        "--ddim_steps",
        "8" "--out",
        "/lam-248-lambdafs/teams/proj-compose/wthomason/lottery/epsilon/robomimic/ticket_{}/{}/outputs",
        "--epsilon",
    ]

    for eps in tqdm(epsilons, desc="Epsilon values", unit="value", leave=True):
        # Format epsilon to avoiding scientific notation (e.g., 0.0000 instead of 0e+00)
        eps_str = f"{eps:.4f}"
        for ticket_idx in tqdm(
            range(num_tickets), desc="Tickets", unit="ticket", leave=False
        ):
            tqdm.write(
                f"\n>>> Running trial for ticket = {ticket_idx} and ticket_epsilon = {eps_str}"
            )
            cmd = base_command.copy()
            cmd[-2] = cmd[-2].format(ticket_idx, eps_str)
            cmd.append(eps_str)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    !! Error: Command failed with exit code {e.returncode}")
                return
            except KeyboardInterrupt:
                print("\n    Script interrupted by user. Exiting...")
                sys.exit(1)


if __name__ == "__main__":
    main()
