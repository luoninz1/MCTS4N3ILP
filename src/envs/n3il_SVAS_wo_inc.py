# No three in line environment with Second Valid Action Space but without incremental update (SVAS wo inc)
from src.envs.collinear_for_mcts import N3il
from src.rewards.n3il_rewards import get_value_nb
import numpy as np
from numba import njit

class N3il_with_SVAS_wo_inc(N3il):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        self.name = 'N3il_with_SVAS_wo_inc'

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state=None):
        # Get all valid moves
        return super().get_valid_moves(super().get_next_state(parent_state.copy(), action_taken))

        