import numpy as np
import random
import time
import warnings
import os
import re

from src.envs import N3il, N3il_with_symmetry, N3il_with_symmetry_and_symmetric_actions, supnorm_priority, supnorm_priority_array
from src.utils.symmetry import get_d4_orbit
from src.algos import MCTS, MCTS_Tree_Reuse, ParallelMCTS, LeafChildParallelMCTS, MCGS

def set_seeds(seed):
    """Set random seeds for reproducibility across all random number generators."""
    np.random.seed(seed)
    random.seed(seed)
    # Force compilation of numba functions with the seeded state
    # This ensures numba's internal random state is also seeded
    from numba import config
    config.THREADING_LAYER = 'safe'

def select_outermost_with_tiebreaker(mcts_probs, n):
    """
    Select an action from the outermost positions among those with the highest MCTS probability.
    If multiple actions have the same max probability and distance to edge, break ties randomly.
    Note: Uses numpy's random number generator which should be seeded for reproducibility.
    """
    # Reshape the 1D probability array to 2D grid
    mcts_probs_2d = mcts_probs.reshape((n, n))
    max_val = np.max(mcts_probs_2d)

    # Find all positions with maximum probability
    max_indices = np.argwhere(mcts_probs_2d == max_val)

    # Define distance to nearest board edge
    def edge_distance(i, j):
        return min(i, n - 1 - i, j, n - 1 - j)

    # Compute edge distance for each candidate
    distances = [edge_distance(i, j) for i, j in max_indices]
    min_dist = min(distances)

    # Select all actions with minimum edge distance
    outermost_positions = [pos for pos, dist in zip(max_indices, distances) if dist == min_dist]

    # Break ties randomly among outermost positions
    chosen_pos = outermost_positions[np.random.choice(len(outermost_positions))]
    action = chosen_pos[0] * n + chosen_pos[1]
    return action

def evaluate(args):
    # Set random seeds for reproducibility at the start of evaluation
    if 'random_seed' in args:
        set_seeds(args['random_seed'])
        # Also warmup numba functions with seeded state
        # dummy_state = np.zeros((2, 2), dtype=np.int8)
        # _ = simulate_nb(dummy_state, 2, 2, 4)
    
    # priority_grid_arr = supnorm_priority_array(args['n']) # This version no longer support priority
    priority_grid_arr = None
    start = time.time()
    n = args['n']

    # Define the environment based on args
    if args['environment'] == 'N3il_with_symmetry':
        n3il =  N3il_with_symmetry(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    elif args['environment'] == 'N3il':
        n3il = N3il(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    elif args['environment'] == 'N3il_with_symmetry_and_symmetric_actions':
        n3il = N3il_with_symmetry_and_symmetric_actions(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    else:
        raise ValueError(f"Unknown environment: {args['environment']}")

    if args['algorithm'] == 'MCGS':
        if args.get('num_workers', 1) == 1:
            mcts_cls = MCGS
        else:
            raise ValueError("MCGS does not support parallel execution yet.")
    elif args['algorithm'] == 'MCTS_Tree_Reuse':
        mcts_cls = MCTS_Tree_Reuse
    elif args['algorithm'] == 'MCTS':
        # Check if leaf/child parallel is requested
        if args.get('child_parallel', False) or args.get('simulations_per_leaf', 1) > 1:
            mcts_cls = LeafChildParallelMCTS
        elif args.get('num_workers', 1) <= 1:
            mcts_cls = MCTS
        else:
            mcts_cls = ParallelMCTS
    elif args['algorithm'] == 'LeafChildParallelMCTS':
        mcts_cls = LeafChildParallelMCTS
    else:
        raise ValueError(f"Unknown algorithm: {args['algorithm']}")
    
    # Initialize MCTS or MCGS
    mcts = mcts_cls(n3il, args=args)
    
    # Set trial ID for tree visualization
    if args.get('tree_visualization', False):
        trial_id = f"trial_{args.get('random_seed', 'unknown')}_n{args.get('n', 'unknown')}"
        mcts.trial_id = trial_id

    # Handle continuation from existing state
    continue_path = args.get('continue_from_existing_state')
    
    if continue_path:
        target_npy_file = None
        
        # If path doesn't exist, raise error immediately
        if not os.path.exists(continue_path):
            raise FileNotFoundError(f"Path for continuation not found: {continue_path}")
            
        if os.path.isdir(continue_path):
            # Find the best .npy file in the directory (max pts, deepest search)
            npy_files = [f for f in os.listdir(continue_path) if f.endswith('.npy')]
            if not npy_files:
                raise FileNotFoundError(f"No .npy files found in directory: {continue_path}")
            
            best_file = None
            max_pts = -1
            
            # Pattern to extract pts count: ..._pts{number}_...
            pattern = re.compile(r'_pts(\d+)_')
            
            for f in npy_files:
                match = pattern.search(f)
                if match:
                    pts = int(match.group(1))
                    # Pick the one with most points (deepest search)
                    if pts > max_pts:
                        max_pts = pts
                        best_file = f
            
            if best_file:
                target_npy_file = os.path.join(continue_path, best_file)
                print(f"Resuming from best found state (pts={max_pts}): {target_npy_file}")
            else:
                 # If no pattern matched, maybe just take the last one alphabetically or raise error?
                 # Let's try to just use the sorted last file as fallback or raise an error
                 raise ValueError(f"Could not determine best state file (with _ptsN_ pattern) from {continue_path}")
        
        elif os.path.isfile(continue_path):
            if not continue_path.endswith('.npy'):
                 raise ValueError(f"Specified file must be an .npy file: {continue_path}")
            target_npy_file = continue_path
            print(f"Resuming from specified state: {target_npy_file}")

        # Final check
        if not target_npy_file or not os.path.exists(target_npy_file):
             raise FileNotFoundError(f"Target state file does not exist: {target_npy_file}")

        # Load state
        try:
            state = np.load(target_npy_file)
            
            # Check shape
            if state.shape != (n, n):
                 raise ValueError(f"Loaded state shape {state.shape} does not match expected grid size ({n}, {n})")

            num_of_points = int(np.sum(state))
            print(f"Successfully loaded state with {num_of_points} points.")
            
            # Check if the loaded state is terminal
            valid_moves_check = n3il.get_valid_moves(state)
            if np.sum(valid_moves_check) == 0:
                warnings.warn(f"Loaded state from {target_npy_file} is terminal. Exiting evaluation.")
                return num_of_points

        except Exception as e:
            raise RuntimeError(f"Failed to load state from {target_npy_file}: {e}")
        
    else:
        state = n3il.get_initial_state()
        num_of_points = 0

    while True:
        if args['display_state'] == True:
            print("---------------------------")
            print(f"Number of points: {num_of_points}")
            print(state)

        valid_moves = n3il.get_valid_moves(state)
        value, is_terminal = n3il.get_value_and_terminated(state, valid_moves)

        if is_terminal:
            print("*******************************************************************")
            print(f"Trial Terminated with {num_of_points} points. Final valid configuration:")
            print(state)
            n3il.display_state(state, mcts_probs)
            end = time.time()
            print(f"Time: {end - start:.6f} sec")
            
            # Record results to table
            n3il.record_to_table(
                terminal_num_points=num_of_points,
                start_time=start,
                end_time=end,
                time_used=end - start
            )
            
            # Save tree visualization snapshots if enabled
            if args.get('tree_visualization', False) and hasattr(mcts, 'snapshots') and mcts.snapshots:
                print(f"Collected {len(mcts.snapshots)} tree snapshots for this trial")
                print(f"Total global snapshots so far: {len(MCTS.global_trial_data)}")
                # Individual HTML files are no longer generated per trial
                # All data is aggregated globally for comprehensive viewing
                
            break

        # Get MCTS probabilities
        mcts_probs = mcts.search(state)

        # Use outermost-priority selector to pick action
        action = select_outermost_with_tiebreaker(mcts_probs, n)

        # Display MCTS probabilities and board
        if args['display_state'] == True:
            n3il.display_state(state, mcts_probs)

        if args.get('symmetric_action', None):
             # Logic is now handled by N3il_with_symmetry_and_symmetric_actions internal get_next_state
             # However, we need to know the number of points added to increment num_of_points correctly.
             
             next_state = n3il.get_next_state(state, action)
             
             # Calculate points added
             prev_count = np.sum(state)
             new_count = np.sum(next_state)
             points_added = new_count - prev_count
             num_of_points += points_added
             
             state = next_state
             
             # Log actions
             if points_added > 1:
                print(f"Applied symmetric batch action: {points_added} points added.")
             else:
                warnings.warn(f"Symmetric action fallback triggered. Only single action {action} applied.")

        else:
            # Apply action
            num_of_points += 1
            state = n3il.get_next_state(state, action)
    
    if args['logging_mode'] == True:
        return num_of_points