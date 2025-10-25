"""
Visualization utilities for No-Three-In-Line environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


def display_state(env, state, action_prob=None):
    """
    Display the current grid configuration using matplotlib.
    Points are drawn where the state equals 1.
    The origin (0, 0) is located at the bottom-left.
    If action_prob is provided (1D array), it is reshaped and overlaid as a heatmap.
    Marker sizes, font sizes, and grid line widths auto-adjust to the grid dimensions.
    """
    rows, cols = env.row_count, env.column_count

    # Create date-time folder once per instance
    if not hasattr(env, '_display_folder'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{rows}by{cols}"
        base_dir = env.args.get('figure_dir', 'figures')
        env._display_folder = os.path.join(base_dir, folder_name)
        os.makedirs(env._display_folder, exist_ok=True)

    # keep a fixed figure size
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # dynamically compute sizes (fixed fig)
    marker_size     = 50000.0 / (rows * cols)      # scatter dot size
    font_size       = 200.0    / max(rows, cols)   # annotation font size
    grid_line_width = 2.0      / max(rows, cols)   # grid line width
    cb_shrink       = 0.8                        # colorbar shrink

    # Calculate proper DPI based on grid size for clarity while maintaining speed
    grid_area = rows * cols
    if grid_area <= 100:      # 10x10 or smaller
        dpi = 150
    elif grid_area <= 400:    # 20x20 or smaller
        dpi = 200
    elif grid_area <= 900:    # 30x30 or smaller
        dpi = 250
    else:                     # larger grids
        dpi = min(300, 150 + (grid_area - 900) // 50)  # gradually increase, cap at 300

    # Calculate title font size - bigger for larger grids
    title_font_size = max(10, min(20, 8 + max(rows, cols) * 0.3))

    if action_prob is not None:
        assert action_prob.shape[0] == rows * cols, \
            f"Expected length {rows * cols}, got {len(action_prob)}"
        action_prob_2d = action_prob.reshape((rows, cols))
        flipped = np.flipud(action_prob_2d)

        im = ax.imshow(
            flipped,
            cmap='Reds',
            alpha=0.6,
            extent=[-0.5, cols - 0.5, -0.5, rows - 0.5],
            origin='lower',
            vmin=0, vmax=action_prob.max() if action_prob.max() > 0 else 1e-5
        )
        plt.colorbar(im, label="Action Probability", shrink=cb_shrink)

        max_val = action_prob_2d.max()
        max_positions = np.argwhere(action_prob_2d == max_val)

        # annotate each cell
        for i in range(rows):
            for j in range(cols):
                val = action_prob_2d[i, j]
                y_disp = rows - 1 - i
                is_max = any((i, j) == tuple(mp) for mp in max_positions)
                color = 'gold' if is_max else ('white' if val > 0.5 * max_val else 'black')
                weight = 'bold' if is_max else 'normal'
                ax.text(
                    j, y_disp, f"{val:.2f}",
                    ha='center', va='center',
                    color=color, weight=weight,
                    fontsize=font_size
                )

    # plot placed points
    y_idx, x_idx = np.nonzero(state)
    y_disp = rows - 1 - y_idx
    ax.scatter(x_idx, y_disp, s=marker_size, c='blue', linewidths=0.5)

    # draw grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=grid_line_width)
    ax.set_xticks([])  # hide tick labels
    ax.set_yticks([])

    # set equal aspect and limits
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')

    # Create comprehensive title with all args info and points count
    num_points = np.sum(state)
    base_title = "No-Three-In-Line Grid with Action Probabilities" if action_prob is not None else "No-Three-In-Line Grid"

    # Extract all key info from args
    algorithm = env.args.get('algorithm', 'Unknown')
    num_searches = env.args.get('num_searches', 'N/A')
    C = env.args.get('C', 'N/A')
    topN = env.args.get('TopN', 'N/A')
    priority = env.args.get('simulate_with_priority', 'N/A')
    workers = env.args.get('num_workers', 'N/A')

    title = f"{base_title}\nGrid: {rows}×{cols}, Points: {num_points}, Algorithm: {algorithm}, Searches: {num_searches}\nC: {C}, TopN: {topN}, Priority: {priority}, Workers: {workers}"
    plt.title(title, fontsize=title_font_size, pad=20)

    plt.tight_layout()

    # Save the plot with timestamp, grid size, and number of points for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"no_three_in_line_{rows}x{cols}_pts{num_points}_{algorithm}_{timestamp}.png"

    # Full path for the file
    full_path = os.path.join(env._display_folder, filename)

    try:
        plt.savefig(full_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as: {full_path} (DPI: {dpi})")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()  # Close the figure to free memory
