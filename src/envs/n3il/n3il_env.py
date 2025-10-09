"""
No-Three-In-Line environment base class.

This module provides the core environment implementation for the
No-Three-In-Line problem, compatible with MCTS algorithms.
"""

import numpy as np
import random
import datetime
from src.algos.mcts.utils import get_valid_moves_nb, get_valid_moves_subset_nb, filter_top_priority_moves
from src.envs.n3il.rewards import get_value_nb
from src.envs.n3il.visualization import display_state
from src.envs.n3il.logging import record_to_table


class N3il:
    """
    Base No-Three-In-Line environment.
    Compatible with MCTS-based algorithms.
    """

    def __init__(self, grid_size, args, priority_grid=None):
        # Set random seed for reproducibility
        if 'random_seed' in args:
            np.random.seed(args['random_seed'])
            random.seed(args['random_seed'])
            # Trigger numba compilation with seeded state (lazy import to avoid circular dependency)
            from src.algos.mcts.simulation import simulate_nb
            _ = simulate_nb(np.zeros((2, 2), dtype=np.int8), 2, 2, 4)

        self.row_count, self.column_count = grid_size
        self.action_size = self.row_count * self.column_count
        self.pts_upper_bound = np.min(grid_size) * 2
        self.priority_grid = priority_grid if priority_grid is not None else np.zeros(grid_size)
        self.args = args

        # Set max_level_to_use_symmetry with default value
        self.max_level_to_use_symmetry = args.get('max_level_to_use_symmetry', 0)

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
        """Get all valid moves, optionally filtered by priority."""
        valid_moves = get_valid_moves_nb(state, self.row_count, self.column_count)

        # Only keep moves with the highest priority if priority grid is provided
        if self.priority_grid is not None and self.args.get('TopN') is not None:
            return filter_top_priority_moves(
                valid_moves,
                self.priority_grid,
                self.row_count,
                self.column_count,
                top_N=self.args['TopN']
            )
        else:
            return valid_moves

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        """Get valid moves for child state, optionally filtered by priority."""
        valid_moves = get_valid_moves_subset_nb(
            parent_state, parent_valid_moves, action_taken,
            self.row_count, self.column_count
        )

        # Only keep moves with the highest priority if priority grid is provided
        if self.priority_grid is not None and self.args.get('TopN') is not None:
            return filter_top_priority_moves(
                valid_moves,
                self.priority_grid,
                self.row_count,
                self.column_count,
                top_N=self.args['TopN']
            )
        else:
            return valid_moves

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
        """Encode state for neural network input (if needed)."""
        encoded_state = np.stack(
            (state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state

    def display_state(self, state, action_prob=None):
        """Display the current grid configuration using matplotlib."""
        display_state(self, state, action_prob)

    def record_to_table(self, terminal_num_points, start_time, end_time, time_used):
        """Record experiment results to a CSV table."""
        record_to_table(self, terminal_num_points, start_time, end_time, time_used)
