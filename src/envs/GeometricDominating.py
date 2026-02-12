"""
FORMAL DEFINITION: Minimum Geometric Dominating Set (D_n)

Let G_n = {1, ..., n} x {1, ..., n} be the n x n grid of lattice points.
A subset S ⊆ G_n is a 'geometric dominating set' if every point P ∈ G_n 
is either:
    1. An element of S (P ∈ S), or
    2. Collinear with at least two distinct points in S.

The quantity D_n is defined as the minimum cardinality of such a set S:
    D_n = min { |S| : S ⊆ G_n is a geometric dominating set }

Current bounds for D_n:
    - Upper Bound: D_n <= 2 * ceil(n/2)
    - Lower Bound: D_n = Ω(n^(2/3))
"""

from src.envs.n3il_symmetric_action import N3il_with_symmetry_and_symmetric_actions, get_d4_orbit_nb
from src.envs.collinear_for_mcts import filter_top_priority_moves
from src.rewards.n3il_rewards import get_value_nb
import numpy as np
from numba import njit
import math

@njit(cache=True)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

@njit(cache=True)
def check_dominating_nb(state, n):
    """
    Checks if the set of points in 'state' is a geometric dominating set.
    Returns True if dominating, False otherwise.
    """
    dominated = state.copy()
    rows, cols = np.where(state == 1)
    k = len(rows)
    
    if k == 0:
        return n == 0
    
    # Mark points on lines formed by pairs of points in S
    for i in range(k):
        r1, c1 = rows[i], cols[i]
        for j in range(i + 1, k):
            r2, c2 = rows[j], cols[j]
            
            dr = r2 - r1
            dc = c2 - c1
            
            g = gcd(abs(dr), abs(dc))
            step_r = dr // g
            step_c = dc // g
            
            # Start from r1, c1 and go positive direction
            cur_r, cur_c = r1 + step_r, c1 + step_c
            while 0 <= cur_r < n and 0 <= cur_c < n:
                dominated[cur_r, cur_c] = 1
                cur_r += step_r
                cur_c += step_c
                
            # Start from r1, c1 and go negative direction
            cur_r, cur_c = r1 - step_r, c1 - step_c
            while 0 <= cur_r < n and 0 <= cur_c < n:
                dominated[cur_r, cur_c] = 1
                cur_r -= step_r
                cur_c -= step_c
                
    # Check if all points are dominated
    total_dominated = np.sum(dominated)
    return total_dominated == n * n

@njit(cache=True)
def get_valid_moves_dominating_nb(state, n):
    # Valid moves are all empty cells
    # Return 1D array where 1=valid, 0=invalid
    return (1 - state.flatten()).astype(np.int8)

@njit(cache=True)
def get_next_state_dominating_nb(state, action, n, op_codes, current_mask):
    """
    Apply symmetric actions for Dominating Set problem.
    Since we don't have constraints like 'No-3-in-line',
    we effectively just add the points in the orbit if they are empty.
    """
    orbit = get_d4_orbit_nb(action, n, op_codes)
    
    if current_mask[action] == 0:
        return state

    temp_state = state.copy()
    temp_mask = current_mask.copy()
    possible = True
    
    for act in orbit:
        if temp_mask[act] == 0:
            possible = False
            break
        temp_mask[act] = 0
        r, c = act // n, act % n
        temp_state[r, c] = 1
        
    if possible:
        return temp_state
    else:
        # Fallback to single action
        res = state.copy()
        res[action // n, action % n] = 1
        return res

@njit(cache=True)
def update_dominating_nb(state, dominated_mask, n, new_action_idx):
    # Updates the dominated_mask given a new point at new_action_idx
    r_new = new_action_idx // n
    c_new = new_action_idx % n
    
    dominated_mask[r_new, c_new] = 1
    
    rows, cols = np.where(state == 1)
    k = len(rows)
    
    for i in range(k):
        r_old = rows[i]
        c_old = cols[i]
        
        # Skip self (point just added is in state now)
        if r_old == r_new and c_old == c_new:
            continue
            
        dr = r_old - r_new
        dc = c_old - c_new
        
        g = gcd(abs(dr), abs(dc))
        step_r = dr // g
        step_c = dc // g
        
        cur_r, cur_c = r_new + step_r, c_new + step_c
        while 0 <= cur_r < n and 0 <= cur_c < n:
            dominated_mask[cur_r, cur_c] = 1
            cur_r += step_r
            cur_c += step_c
            
        cur_r, cur_c = r_new - step_r, c_new - step_c
        while 0 <= cur_r < n and 0 <= cur_c < n:
            dominated_mask[cur_r, cur_c] = 1
            cur_r -= step_r
            cur_c -= step_c
            
    return dominated_mask

@njit(cache=True)
def simulate_dominating_nb(state, n):
    """
    Random rollout until dominating set is formed.
    """
    temp_state = state.copy()
    
    # 1. Compute initial dominated mask
    dominated_mask = temp_state.copy()
    rows, cols = np.where(temp_state == 1)
    k = len(rows)
    
    for i in range(k):
        r1, c1 = rows[i], cols[i]
        for j in range(i + 1, k):
            r2, c2 = rows[j], cols[j]
            dr = r2 - r1
            dc = c2 - c1
            g = gcd(abs(dr), abs(dc))
            sr, sc = dr // g, dc // g
            
            cr, cc = r1 + sr, c1 + sc
            while 0 <= cr < n and 0 <= cc < n:
                dominated_mask[cr, cc] = 1
                cr += sr
                cc += sc
            cr, cc = r1 - sr, c1 - sc
            while 0 <= cr < n and 0 <= cc < n:
                dominated_mask[cr, cc] = 1
                cr -= sr
                cc -= sc
                
    if np.sum(dominated_mask) == n * n:
        # Use 2*n as upper bound for reward calculation
        return get_value_nb(temp_state, 2 * n), temp_state
        
    # 2. Add points
    while True:
        if np.sum(dominated_mask) == n * n:
            break
            
        empty_indices = np.where(temp_state.flatten() == 0)[0]
        if len(empty_indices) == 0:
            break
            
        idx = np.random.choice(empty_indices)
        r, c = idx // n, idx % n
        temp_state[r, c] = 1
        
        update_dominating_nb(temp_state, dominated_mask, n, idx)
        
    return get_value_nb(temp_state, 2 * n), temp_state


class GeometricDominating(N3il_with_symmetry_and_symmetric_actions):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)

    def get_next_state(self, state, action, action_space_of_state=None):
        if self.args.get('symmetric_action', False):
             if action_space_of_state is None:
                 action_space_of_state = get_valid_moves_dominating_nb(state, self.row_count)
             return get_next_state_dominating_nb(state, action, self.row_count, self.op_codes, action_space_of_state)
        else:
             s = state.copy()
             r, c = action // self.column_count, action % self.column_count
             s[r, c] = 1
             return s
        
    def get_valid_moves(self, state):
        valid_moves = get_valid_moves_dominating_nb(state, self.row_count)
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args.get('TopN', 10))
        return valid_moves
        
    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state=None):
        diff = current_state - parent_state
        added_indices = np.argwhere(diff == 1)
        
        valid_moves = parent_valid_moves.copy()
        
        for idx in added_indices:
            r, c = idx
            act = r * self.column_count + c
            valid_moves[act] = 0 # Mark as taken
            
        if self.priority_grid is not None:
             return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args.get('TopN', 10))
                
        return valid_moves

    def simulate(self, state):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        return simulate_dominating_nb(s_arr, self.row_count)
    
    def get_value_and_terminated(self, state, valid_moves):
        if check_dominating_nb(state, self.row_count) or np.sum(valid_moves) == 0:
            return get_value_nb(state, self.pts_upper_bound), True
        return 0.0, False