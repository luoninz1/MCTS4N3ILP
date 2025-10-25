"""
This module defines the N3il_V1 by extending the BaseEnvND class.
It implements the complete version of the No-Three-In-Line environment as in Luoning's branch of RLMath project.
"""

import numpy as np
import random
import datetime
from src.algos.mcts.utils import get_valid_moves_nb, get_valid_moves_subset_nb, filter_top_priority_moves
from src.rewards import get_value_function, get_value_nb  # Centralized rewards
from src.envs.n3il.visualization import display_state
from src.envs.n3il.logging import record_to_table


# This is not gym-compatible, but serves as a base class for MCTS-like algorithms
class N3il_V1():
    def __init__(self, grid_size, args, priority_grid=None):
        # Set random seed for reproducibility
        if 'random_seed' in args:
            np.random.seed(args['random_seed'])
            random.seed(args['random_seed'])
            # Trigger numba compilation with seeded state
            _ = simulate_nb(np.zeros((2, 2), dtype=np.int8), 2, 2, 4)
        
        self.row_count, self.column_count = grid_size
        self.action_size = self.row_count * self.column_count
        self.pts_upper_bound = self.row_count * self.column_count
        self.priority_grid = priority_grid if priority_grid is not None else np.zeros(grid_size)
        self.args = args
        
        # Set max_level_to_use_symmetry with default value
        self.max_level_to_use_symmetry = args.get('max_level_to_use_symmetry', 0)
        self.row_count, self.column_count = grid_size
        self.pts_upper_bound = np.min(grid_size) * 2
        self.action_size = self.row_count * self.column_count
        self.args = args
        self.priority_grid = priority_grid  # Store priority grid
        
        # Create session name with timestamp and grid size
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{timestamp}_{self.row_count}by{self.column_count}"

    def state_to_key(self, state):
        """Convert state to a hashable key for the node registry."""
        return tuple(state.flatten())

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), np.uint8)

    def get_next_state(self, state, action):
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = 1
        return state

    def get_valid_moves(self, state):
        # Get all valid moves
        valid_moves = get_valid_moves_nb(state, self.row_count, self.column_count)
        # Only keep moves with the highest priority
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        # Get all valid moves
        valid_moves = get_valid_moves_subset_nb(parent_state, parent_valid_moves, action_taken, self.row_count, self.column_count)
        # Only keep moves with the highest priority
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves

    def check_collinear(self, state, action=None):
        if action is not None:
            temp_state = state.copy()
            row = action // self.column_count
            col = action % self.column_count
            temp_state[row, col] = 1
        else:
            temp_state = state

        # Call numba-accelerated function
        return check_collinear_nb(temp_state, self.row_count, self.column_count)

    def get_value_and_terminated(self, state, valid_moves):
        """
        Return the normalized value and terminal status of the current state.
        Delegates value calculation to get_value_nb().
        """
        if np.sum(valid_moves) > 0:
            return 0.0, False

        value = get_value_nb(state, self.pts_upper_bound)
        return value, True
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state

    def display_state(self, state, action_prob=None):
        """
        Display the current grid configuration using matplotlib.
        Points are drawn where the state equals 1.
        The origin (0, 0) is located at the bottom-left.
        If action_prob is provided (1D array), it is reshaped and overlaid as a heatmap.
        Marker sizes, font sizes, and grid line widths auto-adjust to the grid dimensions.
        """
        rows, cols = self.row_count, self.column_count

        # Create date-time folder once per instance
        if not hasattr(self, '_display_folder'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{rows}by{cols}"
            base_dir = self.args.get('figure_dir', 'figures')
            self._display_folder = os.path.join(base_dir, folder_name)
            os.makedirs(self._display_folder, exist_ok=True)

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
        algorithm = self.args.get('algorithm', 'Unknown')
        num_searches = self.args.get('num_searches', 'N/A')
        C = self.args.get('C', 'N/A')
        topN = self.args.get('TopN', 'N/A')
        priority = self.args.get('simulate_with_priority', 'N/A')
        workers = self.args.get('num_workers', 'N/A')
        
        title = f"{base_title}\nGrid: {rows}×{cols}, Points: {num_points}, Algorithm: {algorithm}, Searches: {num_searches}\nC: {C}, TopN: {topN}, Priority: {priority}, Workers: {workers}"
        plt.title(title, fontsize=title_font_size, pad=20)
        
        plt.tight_layout()
        
        # Save the plot with timestamp, grid size, and number of points for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"no_three_in_line_{rows}x{cols}_pts{num_points}_{algorithm}_{timestamp}.png"
        
        # Full path for the file
        full_path = os.path.join(self._display_folder, filename)
        
        try:
            plt.savefig(full_path, format='png', dpi=dpi, bbox_inches='tight')
            print(f"Plot saved as: {full_path} (DPI: {dpi})")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()  # Close the figure to free memory

    def record_to_table(self, terminal_num_points, start_time, end_time, time_used):
        """
        Record experiment results to a CSV table in the table_dir directory.
        Creates the CSV file if it doesn't exist, otherwise appends data.
        """
        import csv
        import pandas as pd
        
        # Get table directory and create if it doesn't exist
        table_dir = self.args.get('table_dir', 'results')
        os.makedirs(table_dir, exist_ok=True)
        
        # CSV file path
        csv_file = os.path.join(table_dir, 'experiment_results.csv')
        
        # Prepare data row
        data_row = {}
        
        # Add all args as columns
        for key, value in self.args.items():
            data_row[key] = value
        
        # Add additional required columns
        data_row['terminal_num_points'] = terminal_num_points
        data_row['session_name'] = self.session_name
        data_row['start_time'] = start_time
        data_row['end_time'] = end_time
        data_row['time_used'] = time_used
        
        # Check if CSV file exists
        file_exists = os.path.exists(csv_file)
        
        try:
            if file_exists:
                # Read existing CSV to get all possible columns
                existing_df = pd.read_csv(csv_file)
                all_columns = list(existing_df.columns)
                
                # Check if there are new columns from current data_row
                new_columns = [key for key in data_row.keys() if key not in all_columns]
                
                if new_columns:
                    # Add new columns to existing DataFrame with null values
                    for col in new_columns:
                        existing_df[col] = None
                        all_columns.append(col)
                    
                    # Save the updated DataFrame with new columns
                    existing_df.to_csv(csv_file, index=False)
                
                # Create new row with all columns (fill missing with None)
                new_row = {}
                for col in all_columns:
                    new_row[col] = data_row.get(col, None)
                
                # Append to existing CSV
                new_df = pd.DataFrame([new_row])
                new_df.to_csv(csv_file, mode='a', header=False, index=False)
                
            else:
                # Create new CSV file
                df = pd.DataFrame([data_row])
                df.to_csv(csv_file, index=False)
            
            print(f"Results recorded to: {csv_file}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            # Fallback: try simple CSV writing
            try:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data_row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data_row)
                print(f"Results recorded to: {csv_file} (fallback method)")
            except Exception as e2:
                print(f"Error with fallback CSV writing: {e2}")