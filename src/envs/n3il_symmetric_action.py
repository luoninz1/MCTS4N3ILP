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
from src.envs.collinear_for_mcts import get_valid_moves_nb, get_valid_moves_subset_nb, filter_top_priority_moves, N3il_with_symmetry
from src.utils.symmetry import get_d4_orbit

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
def get_next_state_with_symmetry_logic_nb(state, action, n, op_codes, current_mask):
    orbit = get_d4_orbit_nb(action, n, op_codes)
    
    # We must start with fresh valid moves calc to be safe, 
    # though expensive for just one step, it ensures correctness of the fallback check.
    # current_mask = get_valid_moves_nb(state, n, n) # Now passed as argument
    
    # If the primary action is invalid (shouldn't happen in valid MCTS), raise error
    if current_mask[action] == 0:
        raise ValueError("Invalid action in get_next_state: action is strictly invalid in current mask.")

    temp_state = state.copy()
    temp_mask = current_mask.copy()
    possbile = True
    
    for act in orbit:
        if temp_mask[act] == 0:
            possbile = False
            break
        temp_mask = get_valid_moves_subset_nb(temp_state, temp_mask, act, n, n)
        temp_state[act // n, act % n] = 1
        
    if possbile:
        return temp_state
    else:
        # Fallback
        res = state.copy()
        res[action // n, action % n] = 1
        return res

@njit(cache=True)
def simulate_with_symmetry_logic_nb(state, n, pts_upper_bound, op_codes):
    current_state = state.copy()
    valid_moves = get_valid_moves_nb(current_state, n, n)
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
            temp_mask = get_valid_moves_subset_nb(temp_state, temp_mask, act, n, n)
            temp_state[act // n, act % n] = 1
            
        if orbit_possible:
            current_state = temp_state
            valid_moves = temp_mask
        else:
            # Fallback
            valid_moves = get_valid_moves_subset_nb(current_state, valid_moves, action, n, n)
            current_state[action // n, action % n] = 1
            
        total_valid = np.sum(valid_moves)
        
    return get_value_nb(current_state, pts_upper_bound)


class N3il_with_symmetry_and_symmetric_actions(N3il_with_symmetry):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        if 'symmetric_action' in args:
            self.symmetric_action_mode = args['symmetric_action']
            parts = self.symmetric_action_mode.split('_then_')
            ops = []
            for p in parts:
                if p in STR_TO_OP:
                    ops.append(STR_TO_OP[p])
                else:
                    raise ValueError(f"Unknown symmetry operation: {p}")
            self.op_codes = np.array(ops, dtype=np.int64)
        else:
            raise ValueError("symmetric_action parameter is required.")

    def get_symmetric_actions(self, action):
        if not self.symmetric_action_mode:
            return {action}
        return {
            r * self.column_count + c
            for (r, c) in get_d4_orbit(action, self.row_count, self.symmetric_action_mode)
        }
        
    def get_next_state(self, state, action, action_space_of_state=None):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        if action_space_of_state is None:
            action_space_of_state = get_valid_moves_nb(s_arr, self.row_count, self.column_count)
        new_state = get_next_state_with_symmetry_logic_nb(s_arr, action, self.row_count, self.op_codes, action_space_of_state)
        return new_state

    def simulate(self, state):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        return simulate_with_symmetry_logic_nb(s_arr, self.row_count, self.pts_upper_bound, self.op_codes)

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state):
        """
        Calculate valid moves for the current state given the parent state and valid moves.
        This handles the symmetric batch action case where multiple points might have been added.
        """
        # Calculate the difference to find all points added
        diff = current_state - parent_state
        added_indices = np.argwhere(diff == 1)
        
        # Start with the parent's valid moves
        valid_moves = parent_valid_moves.copy()
        temp_state = parent_state.copy()
        
        # Apply updates for each added point sequentially
        for idx in added_indices:
            r, c = idx
            action = r * self.column_count + c
            
            # Update valid moves subset based on this action
            valid_moves = get_valid_moves_subset_nb(
                temp_state, 
                valid_moves, 
                action, 
                self.row_count, 
                self.column_count
            )
            
            # Update temp state to reflect this added point for the next iteration
            temp_state[r, c] = 1
            
        # Only keep moves with the highest priority if priority grid is used
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves
