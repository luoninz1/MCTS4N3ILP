import numpy as np
from numba import njit
from src.algos.mcts.utils import get_valid_moves_nb, get_valid_moves_subset_nb, filter_top_priority_moves
from src.rewards import point_count_value  # Centralized default value function


@njit(cache=True, nogil=True)
def _simulate_nb_core(state, row_count, column_count):
    """
    Perform random rollout until no valid moves remain.
    Returns the final state (simulation core without value calculation).

    Args:
        state: Current board state (will be modified during simulation)
        row_count: Number of rows
        column_count: Number of columns

    Note: This function uses numba's random number generator which is seeded globally.
    """
    max_size = row_count * column_count
    # Initial valid moves mask
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    total_valid = np.sum(valid_moves)

    while total_valid > 0:
        # Build list of valid actions
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1
        # Randomly select one valid action and place the point
        pick = acts[np.random.randint(0, total_valid)]

        # Incrementally update valid_moves using subset-based filtering
        valid_moves = get_valid_moves_subset_nb(
            state,
            valid_moves,
            pick,
            row_count,
            column_count
        )

        r = pick // column_count
        c = pick % column_count
        state[r, c] = 1  # mark the new point

        total_valid = np.sum(valid_moves)

    return state


def simulate_nb(state, row_count, column_count, pts_upper_bound, value_fn=None):
    """
    Wrapper for simulation that applies value function.

    Args:
        state: Current board state
        row_count: Number of rows
        column_count: Number of columns
        pts_upper_bound: Upper bound for normalization
        value_fn: Optional value function (defaults to point_count_value)

    Returns:
        float: Normalized value
    """
    # Run numba-compiled simulation
    final_state = _simulate_nb_core(state, row_count, column_count)

    # Apply value function (outside numba)
    if value_fn is None:
        return point_count_value(final_state, pts_upper_bound)
    else:
        return value_fn(final_state, pts_upper_bound)


@njit(cache=True, nogil=True)
def _simulate_with_priority_nb_core(state, row_count, column_count, priority_grid, top_N):
    """
    Perform a random rollout that first filters valid moves by priority.
    Returns the final state.

    Args:
        state (np.ndarray): 2D board state.
        row_count (int): Number of rows.
        column_count (int): Number of columns.
        priority_grid (np.ndarray): 2D array of priorities.
        top_N (int): Number of top priority levels to keep.

    Returns:
        np.ndarray: Final state after simulation
    """
    max_size = row_count * column_count

    # Initial valid moves mask
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    # Pre-filter by priority
    valid_moves = filter_top_priority_moves(
        valid_moves, priority_grid, row_count, column_count, top_N
    )
    total_valid = np.sum(valid_moves)

    # Rollout until no moves remain
    while total_valid > 0:
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1

        pick = acts[np.random.randint(0, total_valid)]

        # Update valid moves and state
        valid_moves = get_valid_moves_subset_nb(
            state, valid_moves, pick, row_count, column_count
        )
        state[pick // column_count, pick % column_count] = 1

        # Filter again by priority
        valid_moves = filter_top_priority_moves(
            valid_moves, priority_grid, row_count, column_count, top_N
        )
        total_valid = np.sum(valid_moves)

    return state


def simulate_with_priority_nb(state, row_count, column_count, pts_upper_bound, priority_grid, top_N, value_fn=None):
    """
    Wrapper for priority-based simulation that applies value function.

    Args:
        state (np.ndarray): 2D board state.
        row_count (int): Number of rows.
        column_count (int): Number of columns.
        pts_upper_bound (int): Scoring upper bound.
        priority_grid (np.ndarray): 2D array of priorities.
        top_N (int): Number of top priority levels to keep.
        value_fn: Optional value function (defaults to point_count_value)

    Returns:
        float: Normalized final value.
    """
    # Run numba-compiled simulation
    final_state = _simulate_with_priority_nb_core(state, row_count, column_count, priority_grid, top_N)

    # Apply value function (outside numba)
    if value_fn is None:
        return point_count_value(final_state, pts_upper_bound)
    else:
        return value_fn(final_state, pts_upper_bound)


def _rollout_many(child, R: int):
    """Run child.simulate() R times in the same worker thread; return the list of values."""
    if R <= 1:
        return [child.simulate()]
    vals = []
    for _ in range(R):
        vals.append(child.simulate())
    return vals
