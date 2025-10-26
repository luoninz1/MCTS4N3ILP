"""Experiment runner for MCTS N3IL experiments."""
import time
import numpy as np
from src.envs import N3il, N3il_with_symmetry, supnorm_priority_array
from src.utils.seed import set_seeds, warmup_numba


def select_outermost_with_tiebreaker(mcts_probs, n):
    """
    Select action with highest probability, breaking ties by choosing the outermost position.

    Args:
        mcts_probs: 1D array of action probabilities (length n*n)
        n: Grid size

    Returns:
        int: Selected action index
    """
    max_prob = np.max(mcts_probs)
    candidates = np.where(mcts_probs == max_prob)[0]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-breaker: choose outermost (minimize distance from edge)
    min_edge_dist = float('inf')
    chosen = candidates[0]

    for action in candidates:
        row = action // n
        col = action % n
        # Distance from nearest edge
        edge_dist = min(row, col, n - 1 - row, n - 1 - col)
        if edge_dist < min_edge_dist:
            min_edge_dist = edge_dist
            chosen = action

    return chosen


def run_experiment(args):
    """
    Run a single MCTS experiment with given configuration.

    Args:
        args: Dict containing experiment configuration

    Returns:
        int: Terminal number of points achieved (if logging_mode=True), else None
    """
    # Import here to avoid circular dependencies
    from src.algos.mcts import MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS

    # Set seeds if provided
    # Note: warmup_numba() should be called once at process startup,
    # not per experiment. See baseline_pre_refactor.py for example.
    if 'random_seed' in args:
        set_seeds(args['random_seed'])

    # Build priority grid
    priority_grid_arr = supnorm_priority_array(args['n'])
    start = time.time()
    n = args['n']

    # Instantiate environment
    if args['environment'] == 'N3il_with_symmetry':
        game = N3il_with_symmetry(grid_size=(n, n), args=args, priority_grid=priority_grid_arr)
    elif args['environment'] == 'N3il':
        game = N3il(grid_size=(n, n), args=args, priority_grid=priority_grid_arr)
    else:
        raise ValueError(f"Unknown environment: {args['environment']}")

    # Select algorithm
    if args['algorithm'] == 'MCGS':
        if args.get('num_workers', 1) == 1:
            mcts_cls = MCGS
        else:
            raise ValueError("MCGS does not support parallel execution yet.")
    elif args['algorithm'] == 'MCTS':
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

    # Initialize algorithm
    mcts = mcts_cls(game, args=args)

    # Set trial ID for viz if enabled
    if args.get('tree_visualization', False):
        trial_id = f"trial_{args.get('random_seed', 'unknown')}_n{n}"
        mcts.trial_id = trial_id

    # Main experiment loop
    state = game.get_initial_state()
    num_of_points = 0
    mcts_probs = None  # Initialize for potential use in terminal display

    while True:
        if args.get('display_state', False):
            print("---------------------------")
            print(f"Number of points: {num_of_points}")
            print(state)

        valid_moves = game.get_valid_moves(state)
        value, is_terminal = game.get_value_and_terminated(state, valid_moves)

        if is_terminal:
            print("*******************************************************************")
            print(f"Trial Terminated with {num_of_points} points. Final valid configuration:")
            print(state)

            # Display final state
            if hasattr(game, 'display_state'):
                game.display_state(state, mcts_probs if mcts_probs is not None else None)

            end = time.time()
            print(f"Time: {end - start:.6f} sec")

            # Record to table if method exists
            if hasattr(game, 'record_to_table'):
                game.record_to_table(
                    terminal_num_points=num_of_points,
                    start_time=start,
                    end_time=end,
                    time_used=end - start
                )

            # Save viz if enabled
            if args.get('tree_visualization', False) and hasattr(mcts, 'snapshots') and mcts.snapshots:
                print(f"Collected {len(mcts.snapshots)} tree snapshots for this trial")
                # Note: MCTS.global_trial_data aggregation handled by MCTS class

            break

        # MCTS search
        mcts_probs = mcts.search(state)

        # Action selection
        action = select_outermost_with_tiebreaker(mcts_probs, n)

        # Display if requested
        if args.get('display_state', False) and hasattr(game, 'display_state'):
            game.display_state(state, mcts_probs)

        # Apply action
        num_of_points += 1
        state = game.get_next_state(state, action)

    return num_of_points if args.get('logging_mode', False) else None
