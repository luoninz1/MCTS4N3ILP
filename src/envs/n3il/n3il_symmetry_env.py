"""
No-Three-In-Line environment with D4 symmetry support.

This module extends the base N3il environment with D4 symmetry
group operations for more efficient search.
"""

from src.envs.n3il.n3il_env import N3il
from src.envs.n3il.symmetry import filter_actions_by_stabilizer_nb


class N3il_with_symmetry(N3il):
    """
    N3il enhanced with subgroup-based action filtering.
    Filtering is done by the state stabilizer subgroup G_x
    (one of the 10 subgroups of D4 on squares).
    """

    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        self.max_level_to_use_symmetry = args.get('max_level_to_use_symmetry', 0)
        self.use_symmetry = True if self.max_level_to_use_symmetry > 0 else False

    def get_valid_moves_with_symmetry(self, state):
        """Get valid moves with symmetry filtering applied."""
        # Parent valid moves (already possibly TopN-prioritized)
        action_space = super().get_valid_moves(state)

        return action_space, filter_actions_by_stabilizer_nb(
            action_space, state, self.row_count, self.column_count
        )

    def filter_valid_moves_by_symmetry(self, valid_moves, state):
        """
        Filter the given valid moves by the stabilizer subgroup of the current state.
        This is a convenience method to apply symmetry filtering without needing to
        recompute the state.
        """
        return filter_actions_by_stabilizer_nb(
            valid_moves, state, self.row_count, self.column_count
        )

    def get_valid_moves_subset_with_symmetry(self, parent_state, parent_action_space, action_taken):
        """Get valid moves subset with symmetry filtering applied."""
        action_space = super().get_valid_moves_subset(
            parent_state, parent_action_space, action_taken
        )

        # Build child state to compute stabilizer at the node where these moves apply
        child_state = parent_state.copy()
        r = action_taken // self.column_count
        c = action_taken % self.column_count
        child_state[r, c] = 1

        return action_space, filter_actions_by_stabilizer_nb(
            action_space, child_state, self.row_count, self.column_count
        )
