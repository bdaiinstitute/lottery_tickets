from dataclasses import dataclass
from json import load
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
        eval_info_path = epsilon_dir / "outputs/eval_info.json"
        if not eval_info_path.exists():
            raise RuntimeError(
                f"{results_path} was given as a results path but doesn't have the expected results files!"
            )

        with eval_info_path.open() as eval_info_file:
            eval_info = load(eval_info_file)

        task_rewards = [None] * len(eval_info["per_task"])
        task_successes = [None] * len(eval_info["per_task"])
        for task in eval_info["per_task"]:
            task_id = task["task_id"]
            task_rewards[task_id] = task["sum_rewards"]
            task_successes[task_id] = task["successes"]

        rewards.append(np.array(task_rewards))
        successes.append(np.array(task_successes))

    return TicketResult(
        ticket=ticket_name,
        epsilons=epsilons,
        rewards=np.stack(rewards),
        successes=np.stack(successes),
    )


def plot_ticket_success(axes: list[Axes], results: TicketResult) -> None:
    # Adds a scatter plot of success percentage vs. epsilon value for a given ticket to the given
    # axes
    for idx, ax in enumerate(axes):
      ax.scatter(
          results.epsilons,
          100 * np.count_nonzero(results.successes[:, idx, :], axis=1) / results.successes.shape[-1],
          label=results.ticket,
      )


def plot_ticket_rewards(axes: list[Axes], results: TicketResult) -> None:
    # Adds a scatter plot with standard deviation error bars of reward vs. epsilon value for a given
    # ticket to the given axes
    for idx, ax in enumerate(axes):
      ax.errorbar(
          results.epsilons,
          results.rewards[:, idx, :].mean(axis=1),
          yerr=results.rewards[:, idx, :].std(axis=1),
          label=results.ticket,
      )


def main(results_root: Path, output_dir: Path, n_tasks: int = 10) -> None:
    task_success_plots = []
    task_success_axes = []
    task_reward_plots = []
    task_reward_axes = []
    for _ in range(n_tasks):
        success_plot, success_axes = plt.subplots(tight_layout=True)
        success_axes.title.set_text("Success Rate vs. Epsilon")
        success_axes.set_xlabel("Epsilon")
        success_axes.set_ylabel("Success Percentage")
        success_axes.grid(True)
        task_success_plots.append(success_plot)
        task_success_axes.append(success_axes)

        reward_plot, reward_axes = plt.subplots(tight_layout=True)
        reward_axes.title.set_text("Reward vs. Epsilon")
        reward_axes.set_xlabel("Epsilon")
        reward_axes.set_ylabel("Average Reward")
        reward_axes.grid(True)
        task_reward_plots.append(reward_plot)
        task_reward_axes.append(reward_axes)

    for ticket_dir in results_root.iterdir():
        if (ticket_result := load_ticket_results(ticket_dir)) is not None:
            plot_ticket_success(task_success_axes, ticket_result)
            plot_ticket_rewards(task_reward_axes, ticket_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Add legends to both plots
    for i in range(n_tasks):
      task_success_axes[i].legend()
      task_reward_axes[i].legend()

      # Save the plots
      task_success_plots[i].savefig(output_dir / f"task_{i}_success_plot.pdf", dpi=200)
      task_reward_plots[i].savefig(output_dir / f"task_{i}_reward_plot.pdf", dpi=200)


if __name__ == "__main__":
    tyro.cli(main)
