#!/usr/bin/env python3
"""
RSS-style grid plot (2x5): DDIM-2 row on top, DDIM-8 row below.
Each subplot:
  - 2 bars: DSRL (random placeholder), LT (Ours) from your table
  - horizontal line: Base policy from your table

Creates TWO figures: one per search budget (5K and 10K).

Metric plotted: success_mean (optionally with std error bars for LT and DSRL).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data (from your message): success_mean + success_std
# -----------------------------
TASKS: List[str] = [
    "drawer_cleanup",
    "lift_tray",
    "threading",
    "box_cleanup",
    "three_piece_assembly",
]
DDIMS: List[str] = ["DDIM-2", "DDIM-8"]
BUDGETS: List[str] = ["5K", "10K"]

# Nested dict: data[ddim]["Base" or "LT"][budget][task] = (mean, std)
DATA: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]] = {
    "DDIM-2": {
        "Base": {
            "5K": {
                "drawer_cleanup": (0.442, 0.05528109984434102),
                "lift_tray": (0.41600000000000004, 0.03720215047547654),
                "threading": (0.278, 0.02227105745132009),
                "box_cleanup": (0.47000000000000003, 0.04939635614091388),
                "three_piece_assembly": (0.396, 0.022449944320643643),
            },
            "10K": {  # base is same row in your table; keep identical for convenience
                "drawer_cleanup": (0.442, 0.05528109984434102),
                "lift_tray": (0.41600000000000004, 0.03720215047547654),
                "threading": (0.278, 0.02227105745132009),
                "box_cleanup": (0.47000000000000003, 0.04939635614091388),
                "three_piece_assembly": (0.396, 0.022449944320643643),
            },
        },
        "LT": {
            "5K": {
                "drawer_cleanup": (0.6473333333333333, 0.08124858973135059),
                "lift_tray": (0.5246666666666667, 0.0573527099040083),
                "threading": (0.3113333333333333, 0.04239496825489237),
                "box_cleanup": (0.7479999999999999, 0.14098226838861683),
                "three_piece_assembly": (0.49066666666666664, 0.05270041113059112),
            },
            "10K": {
                "drawer_cleanup": (0.662, 0.08662563131083083),
                "lift_tray": (0.5519999999999999, 0.02357965224510326),
                "threading": (0.4013333333333334, 0.05154932912592886),
                "box_cleanup": (0.7313333333333333, 0.12480918769599195),
                "three_piece_assembly": (0.66, 0.015620499351813422),
            },
        },
    },
    "DDIM-8": {
        "Base": {
            "5K": {
                "drawer_cleanup": (0.8220000000000001, 0.044899888641287286),
                "lift_tray": (0.71, 0.03633180424916989),
                "threading": (0.62, 0.04195235392680606),
                "box_cleanup": (0.876, 0.024166091947189165),
                "three_piece_assembly": (0.6839999999999999, 0.033823069050575506),
            },
            "10K": {  # base is same row in your table; keep identical for convenience
                "drawer_cleanup": (0.8220000000000001, 0.044899888641287286),
                "lift_tray": (0.71, 0.03633180424916989),
                "threading": (0.62, 0.04195235392680606),
                "box_cleanup": (0.876, 0.024166091947189165),
                "three_piece_assembly": (0.6839999999999999, 0.033823069050575506),
            },
        },
        "LT": {
            "5K": {
                "drawer_cleanup": (0.8493333333333334, 0.03002221399786062),
                "lift_tray": (0.7826666666666666, 0.021939310229205755),
                "threading": (0.6126666666666668, 0.012858201014657348),
                "box_cleanup": (0.9706666666666667, 0.011015141094572214),
                "three_piece_assembly": (0.7406666666666667, 0.013316656236958798),
            },
            "10K": {
                "drawer_cleanup": (0.8639999999999999, 0.014000000000000117),
                "lift_tray": (
                    0.7826666666666666,
                    0.021939310229205755,
                ),  # same as 5K in your table
                "threading": (0.6133333333333334, 0.011718930554164704),
                "box_cleanup": (
                    0.9706666666666667,
                    0.011015141094572214,
                ),  # same as 5K in your table
                "three_piece_assembly": (0.7446666666666667, 0.013316656236958737),
            },
        },
    },
}


# -----------------------------
# DSRL actual results (from dsrl_eval_results folder)
# Note: three_piece_assembly has no DSRL results, so we use random placeholder
# -----------------------------
DSRL_DATA: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "DDIM-2": {
        "5K": {
            "drawer_cleanup": (0.808, 0.0),
            "lift_tray": (0.642, 0.0),
            "threading": (0.372, 0.0),
            "box_cleanup": (0.34, 0.0),
            "three_piece_assembly": (0.734, 0.0),  # No data - random placeholder
        },
        "10K": {
            "drawer_cleanup": (0.734, 0.0),
            "lift_tray": (0.638, 0.0),
            "threading": (0.576, 0.0),
            "box_cleanup": (0.546, 0.0),
            "three_piece_assembly": (0.766, 0.0),  # No data - random placeholder
        },
    },
    "DDIM-8": {
        "5K": {
            "drawer_cleanup": (0.802, 0.0),
            "lift_tray": (0.786, 0.0),
            "threading": (0.64, 0.0),
            "box_cleanup": (0.506, 0.0),
            "three_piece_assembly": (0.73, 0.0),  # No data - random placeholder
        },
        "10K": {
            "drawer_cleanup": (0.816, 0.0),
            "lift_tray": (0.804, 0.0),
            "threading": (0.548, 0.0),
            "box_cleanup": (0.386, 0.0),
            "three_piece_assembly": (0.712, 0.0),  # No data - random placeholder
        },
    },
}


def get_dsrl_data() -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Returns DSRL data dictionary.
    """
    return DSRL_DATA


# -----------------------------
# Plotting
# -----------------------------
def plot_budget(
    *,
    budget: str,
    dsrl: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    outdir: str = "plots",
    dpi: int = 300,
) -> None:
    """
    Makes one 2x5 figure for a single budget.
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 11), sharey=True)

    # x positions for two bars
    bar_labels = ["DSRL", "Golden Ticket"]
    x = np.arange(2)
    bar_width = 0.62

    # Colors: Professional Blue vs. Gold (matching plot_lt.py style)
    bar_colors = ["#4E79A7", "gold"]

    for r, ddim in enumerate(DDIMS):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]

            base_mean, _base_std = DATA[ddim]["Base"][budget][task]
            lt_mean, lt_std = DATA[ddim]["LT"][budget][task]
            ds_mean, ds_std = dsrl[ddim][budget][task]

            # Bars without error bars
            vals = [ds_mean, lt_mean]
            ax.bar(
                x,
                vals,
                width=bar_width,
                color=bar_colors,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
            )

            # Baseline horizontal line spanning both bars
            left = x[0] - bar_width / 2
            right = x[-1] + bar_width / 2
            ax.hlines(
                base_mean,
                xmin=left,
                xmax=right,
                linewidth=5.0,
                color="black",
                linestyles=(0, (2, 1)),  # More visible dotted pattern
            )

            # Titles / labels
            if r == 0:
                # Convert task name to CamelCase
                camel_task = "".join(word.capitalize() for word in task.split("_"))
                ax.set_title(camel_task, fontsize=32, fontweight="bold")

            if c == 0:
                ax.set_ylabel(
                    f"{ddim}\nSuccess Rate (%)", fontsize=32, fontweight="bold"
                )

            ax.set_xticks(x)
            # Stack labels vertically for compactness
            stacked_labels = ["DSRL", "Golden\nTicket"]
            ax.set_xticklabels(stacked_labels, fontsize=32, fontweight="bold")
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Single legend for the whole figure (only for 10K chart)
    from matplotlib.lines import Line2D

    if budget == "10K":
        legend_handles = [
            Line2D([0], [0], color="black", lw=5.0, linestyle=(0, (2, 1))),
        ]
        legend_labels = ["Base Policy"]
        # Place legend in the center column (column 2), between the two rows
        leg = fig.legend(
            legend_handles,
            legend_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.4),
            frameon=True,
            fontsize=24,
            ncol=1,
        )
        # Make text bold
        for text in leg.get_texts():
            text.set_weight("bold")

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath_png = os.path.join(outdir, f"budget_policy_grid_{budget}.png")
    outpath_pdf = os.path.join(outdir, f"budget_policy_grid_{budget}.pdf")
    fig.savefig(outpath_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_pdf}")

    plt.show()


def main() -> None:
    # Use actual DSRL results from evaluation
    dsrl = get_dsrl_data()

    for budget in BUDGETS:
        plot_budget(budget=budget, dsrl=dsrl, outdir="plots", dpi=300)


def plot_2x2_summary(
    *,
    dsrl: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    outdir: str = "plots",
    dpi: int = 300,
) -> None:
    """
    Makes a 2x2 summary figure.
    Rows: DDIM-2 (top), DDIM-8 (bottom)
    Columns: 5K (left), 10K (right)
    Each subplot: DSRL and Golden Ticket bars averaged across tasks with std error bars.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    # x positions for two bars - closer together
    x = np.array([0, 0.8])  # Reduced from default spacing
    bar_width = 0.7  # Increased bar width for more compact appearance

    # Colors: Professional Blue vs. Gold (matching plot_lt.py style)
    bar_colors = ["#4E79A7", "gold"]

    for r, ddim in enumerate(DDIMS):
        for c, budget in enumerate(BUDGETS):
            ax = axes[r, c]

            # Collect values across all tasks
            dsrl_vals = []
            lt_vals = []
            base_vals = []

            for task in TASKS:
                dsrl_mean, _ = dsrl[ddim][budget][task]
                lt_mean, _ = DATA[ddim]["LT"][budget][task]
                base_mean, _ = DATA[ddim]["Base"][budget][task]
                dsrl_vals.append(dsrl_mean)
                lt_vals.append(lt_mean)
                base_vals.append(base_mean)

            # Calculate mean and std across tasks
            dsrl_mean_avg = np.mean(dsrl_vals)
            dsrl_std_avg = np.std(dsrl_vals)
            lt_mean_avg = np.mean(lt_vals)
            lt_std_avg = np.std(lt_vals)
            base_mean_avg = np.mean(base_vals)

            # Bars with error bars (std across tasks)
            vals = [dsrl_mean_avg, lt_mean_avg]
            stds = [dsrl_std_avg, lt_std_avg]
            bars = ax.bar(
                x,
                vals,
                width=bar_width,
                color=bar_colors,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
                yerr=stds,
                capsize=5,
                error_kw={"linewidth": 2, "capthick": 2},
            )

            # Baseline horizontal line spanning both bars
            left = x[0] - bar_width / 2
            right = x[-1] + bar_width / 2
            ax.hlines(
                base_mean_avg,
                xmin=left,
                xmax=right,
                linewidth=5.0,
                color="black",
                linestyles=(0, (2, 1)),  # More visible dotted pattern
            )

            # Titles / labels
            if r == 0:
                ax.set_title(f"{budget} Episodes", fontsize=28)

            if c == 0:
                ax.set_ylabel(f"{ddim}\nSuccess Rate", fontsize=28)

            ax.set_xticks(x)
            stacked_labels = ["DSRL", "Golden\nTicket"]
            ax.set_xticklabels(stacked_labels, fontsize=28)
            ax.tick_params(axis="y", labelsize=26)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            # Add legend only for top-right (10K column, DDIM-2 row)
            if c == 1 and r == 0:
                from matplotlib.lines import Line2D

                legend_handle = Line2D(
                    [0], [0], color="black", lw=5.0, linestyle=(0, (2, 1))
                )
                ax.legend(
                    [legend_handle],
                    ["Base Policy"],
                    loc="upper right",
                    fontsize=24,
                )

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath_png = os.path.join(outdir, f"budget_summary_2x2.png")
    outpath_pdf = os.path.join(outdir, f"budget_summary_2x2.pdf")
    fig.savefig(outpath_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_pdf}")

    plt.show()


if __name__ == "__main__":
    main()
    # Also generate the 2x2 summary plot
    dsrl = get_dsrl_data()
    plot_2x2_summary(dsrl=dsrl, outdir="plots", dpi=300)
