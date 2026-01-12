from dataclasses import dataclass
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


def load_ticket_results(results_path: Path) -> TicketResult | None:
    if not results_path.is_dir():
        return None

    ticket_name = results_path.name
    epsilons = []
    rewards = []
    successes = []
    for epsilon_dir in sorted(results_path.iterdir()):
        epsilons.append(float(epsilon_dir.name))
        rewards_path = epsilon_dir / "outputs/rewards.npy"
        if not rewards_path.exists():
            raise RuntimeError(
                f"{results_path} was given as a results path but doesn't have the expected results files!"
            )

        rewards.append(np.load(rewards_path))

        successes_path = epsilon_dir / "outputs/success_rates.npy"
        if not successes_path.exists():
            raise RuntimeError(
                f"{results_path} was given as a results path but doesn't have the expected results files!"
            )

        successes.append(np.load(successes_path))

    return TicketResult(
        ticket=ticket_name,
        epsilons=epsilons,
        rewards=np.stack(rewards),
        successes=np.stack(successes),
    )


def plot_ticket_success(axes: Axes, results: TicketResult) -> None:
    axes.scatter(
        results.epsilons,
        100 * results.successes.mean(axis=1),
        label=results.ticket,
    )


def plot_ticket_rewards(axes: Axes, results: TicketResult) -> None:
    # Adds a scatter plot with standard deviation error bars of reward vs. epsilon value for a given
    # ticket to the given axes
    axes.errorbar(
        results.epsilons,
        results.rewards.mean(axis=1),
        yerr=results.rewards.std(axis=1),
        label=results.ticket,
    )


def main(results_root: Path, output_dir: Path) -> None:
    success_plot, success_axes = plt.subplots(tight_layout=True)
    success_axes.title.set_text("Success Rate vs. Epsilon")
    success_axes.set_xlabel("Epsilon")
    success_axes.set_ylabel("Success Percentage")
    success_axes.grid(True)

    reward_plot, reward_axes = plt.subplots(tight_layout=True)
    reward_axes.title.set_text("Reward vs. Epsilon")
    reward_axes.set_xlabel("Epsilon")
    reward_axes.set_ylabel("Average Reward")

    for ticket_dir in results_root.iterdir():
        if (ticket_result := load_ticket_results(ticket_dir)) is not None:
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
