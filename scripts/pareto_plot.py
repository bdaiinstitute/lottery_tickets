import os
import pickle
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

# --- Apply Style from Paper ---
plt.rcParams.update({"font.size": 18})


def analyze_tickets(base_path, threshold=100.0, max_steps=float("inf")):
    """
    Analyzes ticket directories for success rates and success lengths.
    Produces a 2D scatter plot with Pareto frontier analysis.
    """

    # 1. Gather all numbered subdirectories (tickets)
    try:
        entries = os.listdir(base_path)
        ticket_dirs = [
            d
            for d in entries
            if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()
        ]
        ticket_dirs.sort(key=int)
    except FileNotFoundError:
        print(f"Error: The directory '{base_path}' was not found.")
        return

    if not ticket_dirs:
        print(f"No numbered ticket directories found in {base_path}")
        return

    print(
        f"{'Ticket ID':<10} | {'Success Rate':<12} | {'Avg Steps':<12} | {'Status':<10}"
    )
    print("-" * 55)

    scatter_x = []
    scatter_y = []

    # 2. Process each ticket
    for ticket_id in ticket_dirs[:400]:
        ticket_path = os.path.join(base_path, ticket_id)
        pkl_path = os.path.join(ticket_path, "all_rewards_list.pkl")

        if not os.path.exists(pkl_path):
            continue

        try:
            with open(pkl_path, "rb") as f:
                all_rewards_list = pickle.load(f)
        except Exception as e:
            print(f"Error reading {pkl_path}: {e}")
            continue

        success_count = 0
        success_steps = []
        total_episodes = len(all_rewards_list)

        # 3. Analyze each episode
        for episode_rewards in all_rewards_list:
            rewards_arr = np.array(episode_rewards)
            cumulative_rewards = np.cumsum(rewards_arr)
            qualifying_steps = np.where(cumulative_rewards > threshold)[0]

            if qualifying_steps.size > 0:
                success_count += 1
                steps_to_hit = qualifying_steps[0] + 1
                success_steps.append(steps_to_hit)

        # 4. Calculate Statistics
        success_rate = (success_count / total_episodes) if total_episodes > 0 else 0.0

        if len(success_steps) > 0:
            avg_len = np.mean(success_steps)
            avg_len_str = f"{avg_len:.2f}"

            # Filter based on max_steps argument
            # min length = 100
            if avg_len <= max_steps and avg_len >= 180:
                scatter_x.append(avg_len)
                scatter_y.append(success_rate)
                status = "Included"
            else:
                status = "Filtered"
        else:
            avg_len_str = "N/A"
            status = "No Success"

        print(
            f"{ticket_id:<10} | {success_rate:.1%}       | {avg_len_str:<12} | {status}"
        )

    # 5. Scatter plot with Pareto Frontier
    if scatter_x:
        # Use a similar aspect ratio to your bar chart, slightly taller for scatter
        plt.figure(figsize=(9, 5))

        # --- Identify Pareto Frontier ---
        is_pareto = [True] * len(scatter_x)
        for i in range(len(scatter_x)):
            for j in range(len(scatter_x)):
                if i == j:
                    continue
                # Check if j dominates i (Lower steps is better, Higher rate is better)
                if (scatter_x[j] <= scatter_x[i] and scatter_y[j] >= scatter_y[i]) and (
                    scatter_x[j] < scatter_x[i] or scatter_y[j] > scatter_y[i]
                ):
                    is_pareto[i] = False
                    break

        pareto_x = [scatter_x[i] for i in range(len(scatter_x)) if is_pareto[i]]
        pareto_y = [scatter_y[i] for i in range(len(scatter_y)) if is_pareto[i]]
        normal_x = [scatter_x[i] for i in range(len(scatter_x)) if not is_pareto[i]]
        normal_y = [scatter_y[i] for i in range(len(scatter_y)) if not is_pareto[i]]

        # --- Plotting with Matched Style ---

        # Standard Tickets -> "Base Policy" style (lightcoral)
        plt.scatter(
            normal_x,
            normal_y,
            c="lightcoral",
            s=80,
            alpha=0.8,
            edgecolors="white",
            label="Regular Ticket",
        )

        # Pareto Tickets -> "Golden Ticket" style (gold)
        # We'll skip the scatter for Pareto points since we're using images
        # But keep it hidden for legend purposes
        pareto_scatter = plt.scatter(
            [],
            [],  # Empty for now, just for legend
            c="gold",
            s=100,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Pareto Frontier Ticket",
        )

        # Add golden ticket SVG markers for each Pareto point
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from scipy.ndimage import rotate
        import cairosvg
        import io

        # Convert SVG to PNG in memory
        svg_path = os.path.join(os.path.dirname(__file__), "golden_ticket.svg")
        png_data = cairosvg.svg2png(url=svg_path)
        ticket_img = plt.imread(io.BytesIO(png_data), format="png")

        # Rotate the image by 45 degrees
        ticket_img_rotated = rotate(
            ticket_img, angle=45, reshape=True, mode="constant", cval=0
        )

        for px, py in zip(pareto_x, pareto_y):
            im = OffsetImage(ticket_img_rotated, zoom=0.03)  # Adjust zoom for size
            ab = AnnotationBbox(im, (px, py), frameon=False, zorder=6)
            plt.gca().add_artist(ab)

        # Create custom legend with ticket image
        from matplotlib.legend_handler import HandlerBase

        class ImageHandler(HandlerBase):
            def __init__(self, image):
                super().__init__()
                self.image = image

            def create_artists(
                self,
                legend,
                orig_handle,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            ):
                im = OffsetImage(self.image, zoom=0.015)
                ab = AnnotationBbox(
                    im,
                    (width / 2, height / 2),
                    frameon=False,
                    xycoords=trans,
                    boxcoords="offset points",
                    pad=0,
                )
                return [ab]

        plt.xlabel("Avg Steps to Success")
        plt.ylabel("Success Rate")
        # plt.title("Lottery Tickets Analysis") # Optional: Remove title for cleaner paper look

        # Consistent grid style (optional, removed if you want pure white background like bar chart)
        plt.grid(True, linestyle="--", alpha=0.3)

        # Custom legend with ticket image for Pareto points
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        # Create legend handles
        regular_handle = plt.scatter(
            [], [], c="lightcoral", s=80, alpha=0.8, edgecolors="white"
        )

        # For the ticket, create a small image handle
        ticket_legend_img = OffsetImage(ticket_img_rotated, zoom=0.012)

        plt.legend(
            [regular_handle, ticket_legend_img],
            ["Regular Ticket", "Golden Ticket"],
            loc="upper right",
            framealpha=0.9,
            handler_map={OffsetImage: ImageHandler(ticket_img_rotated)},
        )
        plt.tight_layout()

        output_file = "ticket_scatter_styled.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"\nPlot saved to {output_file}")
    else:
        print("\nNo tickets met the criteria to plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/local/ASURITE/opatil3/src/lottery_tickets/pareto_results",
    )
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--max-steps", type=float, default=220)

    args = parser.parse_args()

    full_path = os.path.abspath(os.path.expanduser(args.path))
    print(f"Analyzing: {full_path}")
    print(f"Max Steps: {args.max_steps}\n")

    analyze_tickets(full_path, args.threshold, args.max_steps)
