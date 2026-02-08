import matplotlib.pyplot as plt
import numpy as np

# ============== ROBOMIMIC PLOT ==============
tasks = ["lift", "can", "square", "transport"]

# DDIM-2 results
ddim2_base = [73.8, 18.4, 21.6, 0.0]
ddim2_base_std = [2.48, 2.87, 3.07, 0.0]
ddim2_lt = [96.7, 68.4, 21.9, 1.8]
ddim2_lt_std = [2.14, 4.66, 1.30, 0.72]

# DDIM-8 results
ddim8_base = [78.4, 42.8, 52.2, 10.8]
ddim8_base_std = [3.93, 4.45, 3.66, 2.23]
ddim8_lt = [95.0, 73.4, 31.7, 14.9]
ddim8_lt_std = [1.20, 6.41, 5.33, 5.69]

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(tasks)) * 1.5
width = 0.30

colors = {
    "ddim2_base": "#a6cee3",
    "ddim2_lt": "#1f78b4",
    "ddim8_base": "#fb9a99",
    "ddim8_lt": "#e31a1c",
}

ax.bar(
    x - 1.5 * width,
    ddim2_base,
    width,
    yerr=ddim2_base_std,
    capsize=3,
    label="DDIM-2 Base",
    color=colors["ddim2_base"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x - 0.5 * width,
    ddim2_lt,
    width,
    yerr=ddim2_lt_std,
    capsize=3,
    label="DDIM-2\nAvg Top-3 LTs",
    color=colors["ddim2_lt"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x + 0.5 * width,
    ddim8_base,
    width,
    yerr=ddim8_base_std,
    capsize=3,
    label="DDIM-8 Base",
    color=colors["ddim8_base"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x + 1.5 * width,
    ddim8_lt,
    width,
    yerr=ddim8_lt_std,
    capsize=3,
    label="DDIM-8\nAvg Top-3 LTs",
    color=colors["ddim8_lt"],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_ylabel("Success Rate (%)", fontsize=32, fontweight="bold")
ax.set_xlabel("Task", fontsize=32, fontweight="bold")
ax.set_title("Robomimic: DDIM-2 vs DDIM-8", fontsize=32, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([t.title() for t in tasks], fontsize=32)
ax.tick_params(axis="y", labelsize=24)
ax.set_ylim(0, 102)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Legend at the top in 2x2 layout
ax.legend(loc="upper right", fontsize=24, frameon=True)

plt.tight_layout()
plt.savefig(
    "/home/local/ASURITE/opatil3/src/lottery_tickets/scripts/plots/ddim2_vs_ddim8.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/home/local/ASURITE/opatil3/src/lottery_tickets/scripts/plots/ddim2_vs_ddim8.pdf",
    bbox_inches="tight",
)
plt.close()

print("Saved Robomimic plot")

# ============== DEXMIMICGEN PLOT ==============
tasks = [
    "drawer_cleanup",
    "lift_tray",
    "threading",
    "box_cleanup",
    "three_piece_assembly",
]
tasks_display = [
    "Drawer\nCleanup",
    "Lift\nTray",
    "Threading",
    "Box\nCleanup",
    "Three Piece\nAssembly",
]

# DDIM-2 results
ddim2_base = [44.2, 41.6, 27.8, 47.0, 39.6]
ddim2_base_std = [5.53, 3.72, 2.23, 4.94, 2.24]
ddim2_lt = [70.7, 51.6, 39.2, 82.3, 58.9]
ddim2_lt_std = [5.16, 2.91, 6.68, 3.21, 8.83]

# DDIM-8 results
ddim8_base = [82.2, 71.0, 62.0, 87.6, 68.4]
ddim8_base_std = [4.49, 3.63, 4.20, 2.42, 3.38]
ddim8_lt = [83.7, 78.7, 60.4, 97.5, 71.7]
ddim8_lt_std = [3.95, 3.10, 2.03, 0.64, 4.00]

fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(tasks)) * 1.5
width = 0.30

ax.bar(
    x - 1.5 * width,
    ddim2_base,
    width,
    yerr=ddim2_base_std,
    capsize=3,
    label="DDIM-2 Base",
    color=colors["ddim2_base"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x - 0.5 * width,
    ddim2_lt,
    width,
    yerr=ddim2_lt_std,
    capsize=3,
    label="DDIM-2\nAvg Top-3 LTs",
    color=colors["ddim2_lt"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x + 0.5 * width,
    ddim8_base,
    width,
    yerr=ddim8_base_std,
    capsize=3,
    label="DDIM-8 Base",
    color=colors["ddim8_base"],
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x + 1.5 * width,
    ddim8_lt,
    width,
    yerr=ddim8_lt_std,
    capsize=3,
    label="DDIM-8\nAvg Top-3 LTs",
    color=colors["ddim8_lt"],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_ylabel("Success Rate (%)", fontsize=32, fontweight="bold")
ax.set_xlabel("Task", fontsize=32, fontweight="bold")
ax.set_title("DexMimicGen: DDIM-2 vs DDIM-8", fontsize=32, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tasks_display, fontsize=32)
ax.tick_params(axis="y", labelsize=24)
ax.set_ylim(0, 102)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Legend at the top in 2x2 layout
# ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=18, frameon=True)

plt.tight_layout()
plt.savefig(
    "/home/local/ASURITE/opatil3/src/lottery_tickets/scripts/plots/ddim2_vs_ddim8_dexmimicgen.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/home/local/ASURITE/opatil3/src/lottery_tickets/scripts/plots/ddim2_vs_ddim8_dexmimicgen.pdf",
    bbox_inches="tight",
)
plt.close()

print("Saved DexMimicGen plot")
