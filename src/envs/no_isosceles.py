import itertools
import numpy as np
# print(np.__version__)
# np.random.seed(0) 
from tqdm import trange
from numba import njit
import sys
import math
from typing import Tuple, List, Set, Callable, NamedTuple, Union, Optional, Iterable, Dict
from sympy import Rational, Integer
from sympy.core.numbers import igcd

from src.rewards.n3il_rewards import get_value_nb
from src.envs.n3il_symmetric_action import N3il_with_symmetry_and_symmetric_actions
from src.envs.collinear_for_mcts import filter_top_priority_moves

# Constants
OP_HORIZONTAL_FLIP = 1
OP_VERTICAL_FLIP = 2
OP_DIAGONAL_FLIP = 3
OP_ANTI_DIAGONAL_FLIP = 4
OP_ROTATION_90 = 5
OP_ROTATION_180 = 6
OP_ROTATION_270 = 7

STR_TO_OP = {
    'horizontal_flip': OP_HORIZONTAL_FLIP,
    'vertical_flip': OP_VERTICAL_FLIP,
    'diagonal_flip': OP_DIAGONAL_FLIP,
    'anti_diagonal_flip': OP_ANTI_DIAGONAL_FLIP,
    'rotation_90': OP_ROTATION_90,
    'rotation_180': OP_ROTATION_180,
    'rotation_270': OP_ROTATION_270
}

@njit(cache=True)
def _apply_transform_nb(r, c, n, op_code):
    if op_code == OP_HORIZONTAL_FLIP: return r, n - 1 - c
    elif op_code == OP_VERTICAL_FLIP: return n - 1 - r, c
    elif op_code == OP_DIAGONAL_FLIP: return c, r
    elif op_code == OP_ANTI_DIAGONAL_FLIP: return n - 1 - c, n - 1 - r
    elif op_code == OP_ROTATION_90: return c, n - 1 - r
    elif op_code == OP_ROTATION_180: return n - 1 - r, n - 1 - c
    elif op_code == OP_ROTATION_270: return n - 1 - c, r
    return r, c

@njit(cache=True)
def get_d4_orbit_nb(action, n, op_codes):
    row = action // n
    col = action % n
    max_pts = n * n
    orbit = np.empty(max_pts, dtype=np.int64)
    orbit[0] = action
    count = 1
    
    # Simple BFS-like expansion
    for op in op_codes:
        limit = count
        for i in range(limit):
            curr = orbit[i]
            nr, nc = _apply_transform_nb(curr // n, curr % n, n, op)
            n_act = nr * n + nc
            found = False
            for k in range(count):
                if orbit[k] == n_act:
                    found = True
                    break
            if not found:
                orbit[count] = n_act
                count += 1
    return orbit[:count]

@njit(cache=True)
def _is_isosceles(r1, c1, r2, c2, r3, c3):
    d1 = (r1 - r2)**2 + (c1 - c2)**2
    d2 = (r2 - r3)**2 + (c2 - c3)**2
    d3 = (r3 - r1)**2 + (c3 - c1)**2
    if d1 == d2 or d2 == d3 or d3 == d1:
        return True
    return False

@njit(cache=True)
def get_valid_moves_isosceles_nb(state, row_count, column_count):
    valid_moves = np.zeros(row_count * column_count, dtype=np.int8)
    
    # Collect occupied points
    points = np.empty((row_count * column_count, 2), dtype=np.int32)
    count = 0
    for r in range(row_count):
        for c in range(column_count):
            if state[r, c] == 1:
                points[count, 0] = r
                points[count, 1] = c
                count += 1
    
    for i in range(row_count * column_count):
        r = i // column_count
        c = i % column_count
        
        if state[r, c] == 1:
            continue
            
        is_valid = True
        for j in range(count):
            p1_r = points[j, 0]
            p1_c = points[j, 1]
            for k in range(j + 1, count):
                p2_r = points[k, 0]
                p2_c = points[k, 1]
                
                if _is_isosceles(r, c, p1_r, p1_c, p2_r, p2_c):
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid:
            valid_moves[i] = 1
            
    return valid_moves

@njit(cache=True)
def get_valid_moves_subset_isosceles_nb(parent_state, parent_valid_moves, action_taken, row_count, column_count):
    # Update valid moves given a new point
    ra = action_taken // column_count
    ca = action_taken % column_count
    
    valid_moves = parent_valid_moves.copy()
    valid_moves[action_taken] = 0
    
    # Collect existing points from parent state
    points = np.empty((row_count * column_count, 2), dtype=np.int32)
    count = 0
    for r in range(row_count):
        for c in range(column_count):
            if parent_state[r, c] == 1:
                points[count, 0] = r
                points[count, 1] = c
                count += 1
                
    # Iterate over currently valid moves to see if they stay valid
    for i in range(row_count * column_count):
        if valid_moves[i] == 1:
            rc = i // column_count
            cc = i % column_count
            
            # Check triples: {candidate, action_taken, P} for all P in existing points
            is_valid = True
            for j in range(count):
                pj_r = points[j, 0]
                pj_c = points[j, 1]
                if _is_isosceles(rc, cc, ra, ca, pj_r, pj_c):
                    is_valid = False
                    break
            
            if not is_valid:
                valid_moves[i] = 0
                
    return valid_moves

@njit(cache=True)
def get_next_state_isosceles_logic_nb(state, action, n, op_codes, current_mask):
    orbit = get_d4_orbit_nb(action, n, op_codes)
    
    # We must start with fresh valid moves calc to be safe, 
    # though expensive for just one step, it ensures correctness of the fallback check.
    # current_mask = get_valid_moves_nb(state, n, n) # Now passed as argument
    
    if current_mask[action] == 0:
        # Invalid action in mask
        pass
        # raise ValueError("Invalid action in get_next_state: action is strictly invalid in current mask.")

    temp_state = state.copy()
    temp_mask = current_mask.copy()
    possible = True
    
    for act in orbit:
        if temp_mask[act] == 0:
            possible = False
            break
        temp_mask = get_valid_moves_subset_isosceles_nb(temp_state, temp_mask, act, n, n)
        temp_state[act // n, act % n] = 1
        
    if possible:
        return temp_state
    else:
        # Fallback
        res = state.copy()
        res[action // n, action % n] = 1
        return res

@njit(cache=True)
def simulate_isosceles_nb(state, n, pts_upper_bound, op_codes):
    current_state = state.copy()
    valid_moves = get_valid_moves_isosceles_nb(current_state, n, n)
    total_valid = np.sum(valid_moves)
    
    while total_valid > 0:
        # 1. Select random valid action
        # Efficient random pick from boolean mask
        # Count non-zeros first, pick Nth, find index
        # (Already calculated total_valid)
        rand_count = np.random.randint(0, total_valid)
        action = -1
        c = 0
        for i in range(n*n):
            if valid_moves[i]:
                if c == rand_count:
                    action = i
                    break
                c += 1
        
        # 2. Try orbit application
        orbit = get_d4_orbit_nb(action, n, op_codes)
        
        temp_state = current_state.copy()
        temp_mask = valid_moves.copy()
        orbit_possible = True
        
        for act in orbit:
            if temp_mask[act] == 0:
                orbit_possible = False
                break
            temp_mask = get_valid_moves_subset_isosceles_nb(temp_state, temp_mask, act, n, n)
            temp_state[act // n, act % n] = 1
            
        if orbit_possible:
            current_state = temp_state
            valid_moves = temp_mask
        else:
            # Fallback
            valid_moves = get_valid_moves_subset_isosceles_nb(current_state, valid_moves, action, n, n)
            current_state[action // n, action % n] = 1
            
        total_valid = np.sum(valid_moves)
        
    return get_value_nb(current_state, pts_upper_bound), current_state

class No_isosceles(N3il_with_symmetry_and_symmetric_actions):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        self.name = "No_isosceles"

    def get_valid_moves(self, state):
        state = np.array(state).reshape(self.row_count, self.column_count)
        valid_moves = get_valid_moves_isosceles_nb(state, self.row_count, self.column_count)
        if self.priority_grid is not None:
             return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        return valid_moves

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state):
        diff = current_state - parent_state
        added_indices = np.argwhere(diff == 1)
        
        valid_moves = parent_valid_moves.copy()
        temp_state = parent_state.copy()
        
        for idx in added_indices:
            r, c = idx
            action = r * self.column_count + c
            
            valid_moves = get_valid_moves_subset_isosceles_nb(
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

    def get_next_state(self, state, action, action_space_of_state=None):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        if action_space_of_state is None:
            action_space_of_state = get_valid_moves_isosceles_nb(s_arr, self.row_count, self.column_count)
            
        new_state = get_next_state_isosceles_logic_nb(s_arr, action, self.row_count, self.op_codes, action_space_of_state)
        return new_state

    def simulate(self, state):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        return simulate_isosceles_nb(s_arr, self.row_count, self.pts_upper_bound, self.op_codes)