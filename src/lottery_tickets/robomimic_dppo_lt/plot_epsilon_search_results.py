from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.axes import Axes
from numpy.typing import NDArray


@dataclass(slots=True)
class TicketResult:
    ticket: str
    epsilons: list[float]
    rewards: NDArray
    successes: NDArray


def load_ticket_results(task_name: str, results_path: Path) -> TicketResult | None:
    if not results_path.is_dir():
        return None

    ticket_name = results_path.name
    epsilons = []
    rewards = []
    successes = []
    for epsilon_dir in sorted(results_path.iterdir()):
        epsilon_results_prefix = list(
            (epsilon_dir / "outputs" / task_name).glob("envs*/eval_*/noise_idx_*/")
        )
        if len(epsilon_results_prefix) < 1:
            print(
                f"Warning! {epsilon_dir} doesn't contain the expected results files; skipping!"
            )
            continue

        assert len(epsilon_results_prefix) == 1
        rewards_path = epsilon_results_prefix[0] / "reward_matrix.npy"
        successes_path = epsilon_results_prefix[0] / "success_matrix.npy"
        if not rewards_path.exists() or not successes_path.exists():
            print(
                f"Warning! {epsilon_dir} doesn't contain the expected results files; skipping!"
            )
            continue

        rewards.append(np.load(rewards_path))
        epsilons.append(float(epsilon_dir.name))
        successes.append(np.load(successes_path))

    if not epsilons:
        raise RuntimeError(
            f"{results_path} was given as a results path but contained no results files!"
        )

    return TicketResult(
        ticket=ticket_name,
        epsilons=epsilons,
        rewards=np.stack(rewards),
        successes=np.stack(successes),
    )


def take_top_tickets(
    ticket_results: list[TicketResult | None], top_n: int, split_percentage: float
) -> list[TicketResult]:
    split_tickets = []
    for ticket_result in ticket_results:
        if ticket_result is None:
            continue

        split_idx = floor(split_percentage * ticket_result.successes.shape[1])
        split_tickets.append(
            (
                ticket_result.successes[:, :split_idx].mean(axis=1),
                TicketResult(
                    ticket_result.ticket,
                    ticket_result.epsilons,
                    ticket_result.rewards[:, split_idx:],
                    ticket_result.successes[:, split_idx:],
                ),
            )
        )

    # NOTE: This takes the top N tickets **for each epsilon value separately** and returns a
    # separated list
    top_results = []
    for i, epsilon in enumerate(split_tickets[0][1].epsilon_values):
        epsilon_best = sorted(split_tickets, key=lambda e: e[0][i])[:top_n]
        print(
            f"{epsilon=}:\n\tSearch: {[e[0][i] for e in epsilon_best]}\n\tEval: {[e[1].successes[i].mean() for e in epsilon_best]}"
        )
        top_results.extend(epsilon_best)

    # TODO: deduplicate

    return top_results


def plot_ticket_success(axes: Axes, results: TicketResult) -> None:
    axes.scatter(
        results.epsilons,
        100 * results.successes.mean(axis=(1, 2)),
        label=results.ticket,
    )


def plot_ticket_rewards(axes: Axes, results: TicketResult) -> None:
    # Adds a scatter plot with standard deviation error bars of reward vs. epsilon value for a given
    # ticket to the given axes
    axes.errorbar(
        results.epsilons,
        results.rewards.mean(axis=(1, 2)),
        yerr=results.rewards.std(axis=(1, 2)),
        label=results.ticket,
    )


def main(
    task_name: str,
    results_root: Path,
    output_dir: Path,
    top_n: int = 10,
    eval_split: float = 0.5,
) -> None:
    success_plot, success_axes = plt.subplots(tight_layout=True)
    success_axes.title.set_text(f"{task_name}: Success Rate vs. Epsilon")
    success_axes.set_xlabel("Epsilon")
    success_axes.set_ylabel("Success Percentage")
    success_axes.grid(True)

    reward_plot, reward_axes = plt.subplots(tight_layout=True)
    reward_axes.title.set_text(f"{task_name}: Reward vs. Epsilon")
    reward_axes.set_xlabel("Epsilon")
    reward_axes.set_ylabel("Average Reward")

    ticket_results = [
        load_ticket_results(task_name, ticket_dir)
        for ticket_dir in (results_root / task_name).iterdir()
    ]
    for ticket_result in take_top_tickets(ticket_results, top_n, eval_split):
        plot_ticket_success(success_axes, ticket_result)
        plot_ticket_rewards(reward_axes, ticket_result)

    # Add legends to both plots
    success_axes.legend()
    reward_axes.legend()

    # Save the plots
    output_dir.mkdir(parents=True, exist_ok=True)
    success_plot.savefig(output_dir / "success_plot.pdf", dpi=200)
    reward_plot.savefig(output_dir / "reward_plot.pdf", dpi=200)


if __name__ == "__main__":
    tyro.cli(main)
