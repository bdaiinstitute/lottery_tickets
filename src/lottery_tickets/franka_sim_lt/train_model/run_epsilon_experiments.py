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
        default="./golden_tickets",
        help="Path to a directory containing saved tickets",
    )

    args = parser.parse_args()
    num_epsilons = int(1.0 / args.step_size) + 1
    epsilons = np.linspace(0, 1, num_epsilons)

    base_command = [
        "python",
        "evaluate.py",
        "evaluation.model_path=checkpoints/fm_seed_1001/checkpoints/fm_policy_final.pt",
        f"evaluation.num_episodes={args.episodes}",
        "hydra.run.dir=/lam-248-lambdafs/teams/proj-compose/wthomason/lottery/epsilon/frankasim/{}/{}/outputs",
    ]

    tickets = [p for p in Path(args.ticket_path).iterdir() if p.suffix == ".pt"]

    for eps in tqdm(epsilons, desc="Epsilon values", unit="value", leave=True):
        # Format epsilon to avoiding scientific notation (e.g., 0.0000 instead of 0e+00)
        eps_str = f"{eps:.4f}"

        for ticket_path in tickets:
            tqdm.write(
                f"\n>>> Running trial for ticket = {ticket_path.name} and ticket_epsilon = {eps_str}"
            )
            cmd = base_command.copy()
            cmd[-1] = cmd[-1].format(ticket_path.name, eps_str)
            cmd.append(f"+ticket_epsilon={eps_str}")
            cmd.append(f"+ticket_path={ticket_path}")

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
