import numpy as np
import time


def evaluate(args):
    """
    Main evaluation function for running MCTS experiments.
    """
    # Import inside function to avoid circular imports
    from src.envs import N3il, N3il_with_symmetry, supnorm_priority_array
    from src.algos.mcts.tree_search import MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS
    from src.algos.mcts.utils import set_seeds, select_outermost_with_tiebreaker
    from src.algos.mcts.simulation import simulate_nb

    # Set random seeds for reproducibility at the start of evaluation
    if 'random_seed' in args:
        set_seeds(args['random_seed'])
        # Also warmup numba functions with seeded state
        dummy_state = np.zeros((2, 2), dtype=np.int8)
        _ = simulate_nb(dummy_state, 2, 2, 4)

    priority_grid_arr = supnorm_priority_array(args['n'])
    start = time.time()
    n = args['n']

    # Define the environment based on args
    if args['environment'] == 'N3il_with_symmetry':
        n3il =  N3il_with_symmetry(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    elif args['environment'] == 'N3il':
        n3il = N3il(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    else:
        raise ValueError(f"Unknown environment: {args['environment']}")

    if args['algorithm'] == 'MCGS':
        if args.get('num_workers', 1) == 1:
            mcts_cls = MCGS
        else:
            raise ValueError("MCGS does not support parallel execution yet.")
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

        # Apply action
        num_of_points += 1
        state = n3il.get_next_state(state, action)

    if args['logging_mode'] == True:
        return num_of_points
