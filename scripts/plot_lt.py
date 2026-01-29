import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup Data ---
suites = {
    "Franka": ["pick_cube"],
    "Robomimic": ["lift", "can", "square", "transport"],
    "Libero": ["spatial", "goal", "object"],
    "DexMimicGen": ["drawer", "lift_tray", "threading", "box_cleanup", "assembly"],
}

# Success Rates (%)
# Note: Data entered exactly in the order of the tasks above
### frankasim data
# 50 env searches
# results in /lam-248-lambdafs/teams/proj-maple/erosen/lottery_tickets/src/lottery_tickets/franka_sim_lt/train_model/outputs
# fm_seed_1001: base: 6%, best ticket: 92% (33)
# fm_seed_1002: base: 82% best ticket: 98% (155)
# fm_seed_1003: base: 26%, best ticket: 96% (71)
# fm_seed_1004: base: 40%, best ticket: 98% (191)
frankasim_base_results = [6, 82, 26, 40]
frankasim_gold_results = [92, 98, 96, 98]
frankasim_base_mean = np.mean(frankasim_base_results)
frankasim_gold_mean = np.mean(frankasim_gold_results)
frankasim_base_std = np.std(frankasim_base_results)
frankasim_gold_std = np.std(frankasim_gold_results)


data = {
    "Franka": {
        "Base Policy": [frankasim_base_mean],
        "Golden Ticket": [frankasim_gold_mean],
    },
    "Robomimic": {
        "Base Policy": [78.4, 42.8, 52.2, 10.7],
        "Golden Ticket": [96.2, 80.8, 32.9, 18.8],
    },
    # Libero Golden Ticket results:
    # object: 84, 86, 100, 100, 100, 100, 98, 100, 100
    # spatial: 82, 96, 92, 100, 84, 76, 100, 96, 86, 78
    # goal: 84, 100, 94, 58, 100, 82, 94, 100, 100, 76
    # Libero Base Policy results:
    # spatial: 68.8, 86.2, 92.5, 68.8, 86.2, 50.0, 85.0, 88.8, 80.6, 82.5
    # goal: 72.0, 94.0, 86.0, 52.0, 92.0, 78.0, 80.0, 100.0, 94.0, 68.0
    # object: 82.0, 98.0, 98.0, 98.0, 76.0, 82.0, 100.0, 90.0, 96.0, 100.0
    "Libero": {
        "Base Policy": [
            np.mean(
                [68.8, 86.2, 92.5, 68.8, 86.2, 50.0, 85.0, 88.8, 80.6, 82.5]
            ),  # spatial
            np.mean(
                [72.0, 94.0, 86.0, 52.0, 92.0, 78.0, 80.0, 100.0, 94.0, 68.0]
            ),  # goal
            np.mean(
                [82.0, 98.0, 98.0, 98.0, 76.0, 82.0, 100.0, 90.0, 96.0, 100.0]
            ),  # object
        ],
        "Golden Ticket": [
            np.mean([82, 96, 92, 100, 84, 76, 100, 96, 86, 78]),  # spatial
            np.mean([84, 100, 94, 58, 100, 82, 94, 100, 100, 76]),  # goal
            np.mean([84, 86, 100, 100, 100, 100, 98, 100, 100]),  # object
        ],
    },
    "DexMimicGen": {
        "Base Policy": [82.2, 71.0, 62.0, 87.6, 68.3],
        "Golden Ticket": [88.0, 79.9, 60.0, 97.8, 75.6],
    },
}

data_std = {
    "Franka": {
        "Base Policy": [frankasim_base_std],
        "Golden Ticket": [frankasim_gold_std],
    },
    "Robomimic": {
        "Base Policy": [003.9, 004.4, 003.6, 002.2],
        "Golden Ticket": [001.4, 004.6, 006.7, 003.6],
    },
    "Libero": {
        "Base Policy": [
            np.std(
                [68.8, 86.2, 92.5, 68.8, 86.2, 50.0, 85.0, 88.8, 80.6, 82.5]
            ),  # spatial
            np.std(
                [72.0, 94.0, 86.0, 52.0, 92.0, 78.0, 80.0, 100.0, 94.0, 68.0]
            ),  # goal
            np.std(
                [82.0, 98.0, 98.0, 98.0, 76.0, 82.0, 100.0, 90.0, 96.0, 100.0]
            ),  # object
        ],
        "Golden Ticket": [
            np.std([82, 96, 92, 100, 84, 76, 100, 96, 86, 78]),  # spatial
            np.std([84, 100, 94, 58, 100, 82, 94, 100, 100, 76]),  # goal
            np.std([84, 86, 100, 100, 100, 100, 98, 100, 100]),  # object
        ],
    },
    "DexMimicGen": {
        "Base Policy": [004.4, 003.9, 004.1, 002.4, 003.3],
        "Golden Ticket": [002.9, 004.6, 004.9, 001.1, 003.1],
    },
}


# --- 2. Create Figure ---
# width_ratios ensures the subplots are sized relative to number of tasks
# (14, 2.4) is a very wide, short aspect ratio suitable for top-of-page
fig, axes = plt.subplots(
    1,
    4,
    figsize=(14, 2.4),
    gridspec_kw={"width_ratios": [1, 4, 2, 5]},
    constrained_layout=True,
)

# Colors: Professional Blue vs. Orange
colors = ["#4E79A7", "gold"]
labels = ["Base Policy", "Golden Ticket"]
width = 0.35  # Width of the bars

# --- 3. Plotting Loop ---
for i, (suite_name, tasks) in enumerate(suites.items()):
    ax = axes[i]
    x = np.arange(len(tasks))

    # Get values
    vals_base = data[suite_name]["Base Policy"]
    vals_gold = data[suite_name]["Golden Ticket"]

    # Get Std Dev Values
    err_base = data_std[suite_name]["Base Policy"]
    err_gold = data_std[suite_name]["Golden Ticket"]

    # Plot Bars with Error Bars (yerr)
    # capsize controls the width of the horizontal lines at the top of the error bar
    ax.bar(
        x - width / 2,
        vals_base,
        width,
        yerr=err_base,
        capsize=3,
        label=labels[0],
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(lw=1, capthick=1, ecolor="black"),
    )

    ax.bar(
        x + width / 2,
        vals_gold,
        width,
        yerr=err_gold,
        capsize=3,
        label=labels[1],
        color=colors[1],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(lw=1, capthick=1, ecolor="black"),
    )

    # Formatting
    ax.set_title(
        suite_name, fontsize=12, fontweight="bold", pad=10
    )  # Suite Name at top
    ax.set_xticks(x)

    # Clean Task Labels (replace underscores with newlines for compactness)
    clean_labels = [
        t.replace("_", "\n").replace("cleanup", "clean").title() for t in tasks
    ]
    ax.set_xticklabels(clean_labels, fontsize=12)

    # Y-Axis Polish
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Only show Y-axis label on the very first plot (Franka) to save space
    if i == 0:
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    else:
        ax.set_yticklabels([])  # Hide numbers
        ax.tick_params(axis="y", length=0)  # Hide ticks

    # Add legend inside Robomimic (index 1)
    if i == 1:
        ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.9)

# Save
plt.savefig("success_rates_wide.pdf", bbox_inches="tight")
plt.savefig("success_rates_wide.png", bbox_inches="tight", dpi=300)
