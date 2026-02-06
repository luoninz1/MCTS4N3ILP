from src.envs.collinear_for_mcts import N3il
from src.rewards.n3il_rewards import get_value_for_FVAS_nb
import numpy as np
from numba import njit

# JIT-compiled function to check if three points are collinear
@njit(cache=True)
def _are_collinear(x1, y1, x2, y2, x3, y3):
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

# JIT-compiled function to check if any collinear triples exist on the board
@njit(cache=True, nogil=True)
def exist_collinear_nb(state, row_count, column_count):
    max_pts = row_count * column_count
    coords = np.empty((max_pts, 2), np.int64)
    n_pts = 0

    # Collect all placed point coordinates
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] == 1:
                coords[n_pts, 0] = i
                coords[n_pts, 1] = j
                n_pts += 1

    # Count all collinear triplets
    for a in range(n_pts):
        for b in range(a + 1, n_pts):
            for c in range(b + 1, n_pts):
                i1, j1 = coords[a, 0], coords[a, 1]
                i2, j2 = coords[b, 0], coords[b, 1]
                i3, j3 = coords[c, 0], coords[c, 1]
                if _are_collinear(j1, i1, j2, i2, j3, i3):
                    return True
    return False

# JIT-compiled function to simulate a random rollout
@njit(cache=True, nogil=True)
def simulate_nb(state, row_count, column_count, pts_upper_bound):
    # Prepare valid moves and existing points
    max_pts = row_count * column_count
    valid_moves = np.empty((max_pts, 2), dtype=np.int64)
    coords = np.empty((max_pts, 2), dtype=np.int64)
    
    n_valid = 0
    n_pts = 0
    
    # Collect valid moves (0s) and existing points (1s)
    for r in range(row_count):
        for c in range(column_count):
            if state[r, c] == 0:
                valid_moves[n_valid, 0] = r
                valid_moves[n_valid, 1] = c
                n_valid += 1
            elif state[r, c] == 1:
                coords[n_pts, 0] = r
                coords[n_pts, 1] = c
                n_pts += 1
    
    # Shuffle valid moves (Fisher-Yates)
    for i in range(n_valid - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        # Swap
        r_temp, c_temp = valid_moves[i, 0], valid_moves[i, 1]
        valid_moves[i, 0] = valid_moves[j, 0]
        valid_moves[i, 1] = valid_moves[j, 1]
        valid_moves[j, 0] = r_temp
        valid_moves[j, 1] = c_temp
        
    # Simulation loop
    for k in range(n_valid):
        # Pick next move
        r = valid_moves[k, 0]
        c = valid_moves[k, 1]
        
        # Apply move
        state[r, c] = 1
        
        # Check termination (collinearity involving new point)
        found_line = False
        for a in range(n_pts):
            for b in range(a + 1, n_pts):
                # Check triplet: existing 'a', existing 'b', new 'c'
                # Note: _are_collinear takes (x1, y1, x2, y2, x3, y3)
                if _are_collinear(coords[a, 1], coords[a, 0], 
                                  coords[b, 1], coords[b, 0], 
                                  c, r):
                    found_line = True
                    break
            if found_line:
                break
        
        if found_line:
            # Revert action to return a valid state
            state[r, c] = 0
            # Game Over - return value of current state
            return get_value_for_FVAS_nb(state, pts_upper_bound), state
            
        # Else, accept move and add to coords for next checks
        coords[n_pts, 0] = r
        coords[n_pts, 1] = c
        n_pts += 1
        
    return get_value_for_FVAS_nb(state, pts_upper_bound), state

# Naive Baseline with Fast Valid Action Space (FVAS)
# FVAS: All empty (=0) spots are valid.

class N3il_with_FVAS(N3il):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        self.name = 'N3il_with_FVAS'

    def get_valid_moves(self, state):
        # Get all valid moves
        return state ^ 1

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state=None):
        # Get all valid moves
        return super().get_next_state(parent_state.copy(), action_taken) ^ 1
    
    def get_value_and_terminated(self, state, valid_moves):
        """
        Return the normalized value and terminal status of the current state.
        Delegates value calculation to get_value_for_FVAS_nb().
        """
        if exist_collinear_nb(state, self.row_count, self.column_count):
            value = get_value_for_FVAS_nb(state, self.pts_upper_bound)
            return value, True

        return 0.0, False
    
    def simulate(self, state):
        return simulate_nb(
                state.copy(),
                self.row_count,
                self.column_count,
                self.pts_upper_bound
            )

        