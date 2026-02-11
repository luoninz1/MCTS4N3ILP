import itertools
import numpy as np
from numba import njit
from src.envs.n3il_symmetric_action import N3il_with_symmetry, N3il_with_symmetry_and_symmetric_actions, get_d4_orbit_nb
from src.rewards.n3il_rewards import get_value_nb
from src.envs.collinear_for_mcts import filter_top_priority_moves

@njit(cache=True)
def _are_collinear_nb(c1, r1, c2, r2, c3, r3):
    return (r2 - r1) * (c3 - c2) == (r3 - r2) * (c2 - c1)

@njit(cache=True)
def invalidate_line_nb(mask, r0, c0, dr, dc, n, m):
    # Pre-simplify dr, dc
    a, b = abs(dr), abs(dc)
    while b:
        a, b = b, a % b
    common = a
    if common == 0: return # Should not happen if points are distinct
    dr //= common
    dc //= common
    
    # Forward
    curr_r, curr_c = r0, c0
    while 0 <= curr_r < n and 0 <= curr_c < m:
        mask[curr_r * m + curr_c] = 0
        curr_r += dr
        curr_c += dc
        
    # Backward
    curr_r, curr_c = r0 - dr, c0 - dc
    while 0 <= curr_r < n and 0 <= curr_c < m:
        mask[curr_r * m + curr_c] = 0
        curr_r -= dr
        curr_c -= dc

@njit(cache=True)
def get_valid_moves_n4il_nb(state, n, m):
    # state: n x m
    mask = np.ones(n * m, dtype=np.int8)
    
    # Mark occupied as 0
    occupied_indices = []
    # Avoid flatten copy if possible, but flat iteration is easier
    flat_state = state.flatten()
    for i in range(n * m):
        if flat_state[i] == 1:
            mask[i] = 0
            occupied_indices.append(i)
            
    num_occ = len(occupied_indices)
    
    # Iterate all triplets
    # If a triplet is collinear, invalidate the line
    for i in range(num_occ):
        idx1 = occupied_indices[i]
        r1, c1 = idx1 // m, idx1 % m
        for j in range(i + 1, num_occ):
            idx2 = occupied_indices[j]
            r2, c2 = idx2 // m, idx2 % m
            for k in range(j + 1, num_occ):
                idx3 = occupied_indices[k]
                r3, c3 = idx3 // m, idx3 % m
                
                if _are_collinear_nb(c1, r1, c2, r2, c3, r3):
                    # Found 3 collinear points.
                    # Invalidate the line defined by r1,c1 and r2,c2
                    invalidate_line_nb(mask, r1, c1, r2 - r1, c2 - c1, n, m)
                    
    return mask

@njit(cache=True)
def get_valid_moves_subset_n4il_nb(state, current_mask, action, n, m):
    # state has old points.
    # action is the new point being added.
    mask = current_mask.copy()
    
    new_r, new_c = action // m, action % m
    mask[action] = 0
    
    points_r = []
    points_c = []
    for r in range(n):
        for c in range(m):
            if state[r, c] == 1:
                points_r.append(r)
                points_c.append(c)
                
    num_occ = len(points_r)
    
    # Iterate pairs of existing points
    for i in range(num_occ):
        r1, c1 = points_r[i], points_c[i]
        for j in range(i + 1, num_occ):
            r2, c2 = points_r[j], points_c[j]
            
            # Check if new point is collinear with this pair
            if _are_collinear_nb(c1, r1, c2, r2, new_c, new_r):
                # We found a triplet (p1, p2, new_point).
                # Invalidate the line.
                invalidate_line_nb(mask, new_r, new_c, r1 - new_r, c1 - new_c, n, m)
                
    return mask

@njit(cache=True)
def get_next_state_with_symmetry_logic_n4il_nb(state, action, n, op_codes, current_mask):
    orbit = get_d4_orbit_nb(action, n, op_codes)
    
    if current_mask[action] == 0:
        # In valid MCTS usage this shouldn't happen, but just in case
        pass
        # raise ValueError("Invalid action in get_next_state")

    temp_state = state.copy()
    temp_mask = current_mask.copy()
    possible = True
    
    for act in orbit:
        if temp_mask[act] == 0:
            possible = False
            break
        temp_mask = get_valid_moves_subset_n4il_nb(temp_state, temp_mask, act, n, n)
        temp_state[act // n, act % n] = 1
        
    if possible:
        return temp_state
    else:
        # Fallback to single point if symmetry fails (though usually we enforce symmetry)
        res = state.copy()
        res[action // n, action % n] = 1
        return res

@njit(cache=True)
def simulate_with_symmetry_logic_n4il_nb(state, n, pts_upper_bound, op_codes):
    current_state = state.copy()
    valid_moves = get_valid_moves_n4il_nb(current_state, n, n)
    total_valid = np.sum(valid_moves)
    
    while total_valid > 0:
        # Pick random
        rand_count = np.random.randint(0, total_valid)
        action = -1
        c = 0
        for i in range(n*n):
            if valid_moves[i]:
                if c == rand_count:
                    action = i
                    break
                c += 1
        
        orbit = get_d4_orbit_nb(action, n, op_codes)
        
        temp_state = current_state.copy()
        temp_mask = valid_moves.copy()
        orbit_possible = True
        
        for act in orbit:
            if temp_mask[act] == 0:
                orbit_possible = False
                break
            temp_mask = get_valid_moves_subset_n4il_nb(temp_state, temp_mask, act, n, n)
            temp_state[act // n, act % n] = 1
            
        if orbit_possible:
            current_state = temp_state
            valid_moves = temp_mask
        else:
            valid_moves = get_valid_moves_subset_n4il_nb(current_state, valid_moves, action, n, n)
            current_state[action // n, action % n] = 1
            
        total_valid = np.sum(valid_moves)
        
    return get_value_nb(current_state, pts_upper_bound), current_state

class N4il(N3il_with_symmetry_and_symmetric_actions):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        pass

    def get_next_state(self, state, action, action_space_of_state=None):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        if action_space_of_state is None:
            action_space_of_state = get_valid_moves_n4il_nb(s_arr, self.row_count, self.column_count)
        new_state = get_next_state_with_symmetry_logic_n4il_nb(
            s_arr, action, self.row_count, self.op_codes, action_space_of_state
        )
        return new_state

    def simulate(self, state):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        return simulate_with_symmetry_logic_n4il_nb(s_arr, self.row_count, self.pts_upper_bound, self.op_codes)

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state):
        diff = current_state - parent_state
        added_indices = np.argwhere(diff == 1)
        
        valid_moves = parent_valid_moves.copy()
        temp_state = parent_state.copy()
        
        for idx in added_indices:
            r, c = idx
            action = r * self.column_count + c
            
            valid_moves = get_valid_moves_subset_n4il_nb(
                temp_state, 
                valid_moves, 
                action, 
                self.row_count, 
                self.column_count
            )
            temp_state[r, c] = 1
            
        if self.priority_grid is not None:
             return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves
