import numpy as np
from numba import njit
import random


def set_seeds(seed):
    """Set random seeds for reproducibility across all random number generators."""
    np.random.seed(seed)
    random.seed(seed)
    from numba import config
    config.THREADING_LAYER = 'safe'


@njit(cache=True, nogil=True)
def exploration_decay_nb(x):
    """Monotone-down decay from (0,1) to (1,0)"""
    # Cosine decay
    # return (np.cos(np.pi * x)+1)/2

    # Linear
    # return 1 - 0.7 * x
    # return 1 - x

    # Square root (gentle early decay)
    # return 1 - 0.9 * np.sqrt(x)
    # return 1 - 1 * np.sqrt(x)
    # return 1 - 0.5 * np.sqrt(x)
    # return 1 - 0.7 * np.sqrt(x)
    # return 1 - 0.8 * np.sqrt(x)
    return 1 - 0.85 * np.sqrt(x)

    # Quadratic (faster decay)
    # return 1 - (x ** 2)

    # Exponential (custom normalization)
    # return ((np.exp(1)/(np.exp(1)-1))**2) * ((np.exp(-x)-np.exp(-1)) ** 2)

    # Exponential fast (k=3)
    #k = 3.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k))

    # Exponential slow (k=1)
    # k = 1.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k))

    # Cosine decay
    # return 0.5 * (1 + np.cos(np.pi * x))

    # Rational decay
    # a = 1.0
    # return (1 - x) / (1 + a * x)

    # Logistic decay
    # k = 10.0
    # g0 = 1 / (1 + np.exp(k * (0 - 0.5)))
    # g1 = 1 / (1 + np.exp(k * (1 - 0.5)))
    # gx = 1 / (1 + np.exp(k * (x - 0.5)))
    # return (gx - g1) / (g0 - g1)

    # Cubic decay
    # return 1 - x ** 3
    # return 1 - (0.9 * (x ** 3))


@njit(cache=True, nogil=True)
def value_fn_nb(x):
    # return x
    # return np.exp(x)
    return x


@njit(cache=True, nogil=True)
def _are_collinear(x1, y1, x2, y2, x3, y3):
    """Check if three points are collinear"""
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)


@njit(cache=True, nogil=True)
def get_valid_moves_nb(state, row_count, column_count):
    """Determine valid moves on the board"""
    max_pts = row_count * column_count
    coords = np.empty((max_pts, 2), np.int64)
    n_pts = 0

    # Collect coordinates of existing points
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] == 1:
                coords[n_pts, 0] = i
                coords[n_pts, 1] = j
                n_pts += 1

    mask = np.zeros(row_count * column_count, np.uint8)

    # Check each empty cell
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] != 0:
                continue
            valid = True
            # Check for collinearity with every pair of existing points
            for p in range(n_pts):
                for q in range(p + 1, n_pts):
                    i1, j1 = coords[p, 0], coords[p, 1]
                    i2, j2 = coords[q, 0], coords[q, 1]
                    if _are_collinear(j1, i1, j2, i2, j, i):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                mask[i * column_count + j] = 1
    return mask


@njit(cache=True, nogil=True)
def get_valid_moves_subset_nb(parent_state, parent_valid_moves, action_taken, row_count, column_count):
    """
    Given a parent state (2D boolean array) and its valid move mask (1D uint8 array),
    return a refined valid move mask for the child:
      1) Remove the action just taken.
      2) For each existing point in state, compute the line to the new point,
         then invalidate any intermediate grid points that lie exactly on that line.
      3) If slope is infinite, invalidate entire column; if slope is zero, invalidate entire row.
    Returns a flattened uint8 array: 1 = valid, 0 = invalid.
    """
    # Copy input mask and remove the taken action
    mask = parent_valid_moves.copy()
    mask[action_taken] = 0

    # Coordinates of the newly placed point
    new_r = action_taken // column_count
    new_c = action_taken % column_count

    # Iterate over all existing points
    for pr in range(row_count):
        for pc in range(column_count):
            if not parent_state[pr, pc]:
                continue
            # Skip the new point itself
            if pr == new_r and pc == new_c:
                continue

            dr = pr - new_r
            dc = pc - new_c

            # Infinite slope (vertical line): invalidate entire column
            if dc == 0:
                for rr in range(row_count):
                    idx = rr * column_count + new_c
                    mask[idx] = 0
                continue

            # Zero slope (horizontal line): invalidate entire row
            if dr == 0:
                row_index = pr
                base = row_index * column_count
                for cc in range(column_count):
                    mask[base + cc] = 0
                continue

            # General (non-vertical, non-horizontal) case: remove every point on the infinite line
            # through (new_r,new_c) and (pr,pc), including both the segment and its extensions.
            for cc in range(column_count):
                # compute how far horizontally from the new point
                num = (cc - new_c) * dr
                # only those aligning to integer row are collinear
                if num % dc != 0:
                    continue
                rr = new_r + num // dc
                # skip anything outside the grid
                if rr < 0 or rr >= row_count:
                    continue
                idx = rr * column_count + cc
                mask[idx] = 0

    return mask


@njit(cache=True, nogil=True)
def check_collinear_nb(state, row_count, column_count):
    """Count collinear triples on the board"""
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

    triples = 0
    # Count all collinear triplets
    for a in range(n_pts):
        for b in range(a + 1, n_pts):
            for c in range(b + 1, n_pts):
                i1, j1 = coords[a, 0], coords[a, 1]
                i2, j2 = coords[b, 0], coords[b, 1]
                i3, j3 = coords[c, 0], coords[c, 1]
                if _are_collinear(j1, i1, j2, i2, j3, i3):
                    triples += 1
    return triples


@njit(cache=True, nogil=True)
def filter_top_priority_moves(valid_moves, priority_grid, row_count, column_count, top_N=1):
    """
    Numba-accelerated: Filter valid moves to only those with the top_N highest priorities.

    Args:
        valid_moves (np.ndarray): 1D array (flattened) of valid moves (1=valid, 0=invalid).
        priority_grid (np.ndarray): 2D array of priority values for each grid cell.
        row_count (int): Number of rows in the grid.
        column_count (int): Number of columns in the grid.
        top_N (int): Number of top priority levels to select.

    Returns:
        np.ndarray: 1D mask array with only the top_N-priority valid moves set to 1.
    """
    indices = []
    priorities = []
    for idx in range(valid_moves.shape[0]):
        if valid_moves[idx] == 1:
            indices.append(idx)
            i = idx // column_count
            j = idx % column_count
            priorities.append(priority_grid[i, j])
    if len(indices) == 0:
        return valid_moves

    # Find the unique priorities and sort descending
    # Numba doesn't support np.unique or sort for lists, so do it manually
    # 1. Copy priorities to a new array
    n = len(priorities)
    unique_priorities = []
    for k in range(n):
        p = priorities[k]
        found = False
        for l in range(len(unique_priorities)):
            if unique_priorities[l] == p:
                found = True
                break
        if not found:
            unique_priorities.append(p)
    # 2. Sort unique_priorities descending (simple selection sort)
    for i in range(len(unique_priorities)):
        max_idx = i
        for j in range(i+1, len(unique_priorities)):
            if unique_priorities[j] > unique_priorities[max_idx]:
                max_idx = j
        # Swap
        tmp = unique_priorities[i]
        unique_priorities[i] = unique_priorities[max_idx]
        unique_priorities[max_idx] = tmp

    # 3. Select top_N priorities
    N = min(top_N, len(unique_priorities))
    threshold = unique_priorities[:N]

    # 4. Build mask
    mask = np.zeros_like(valid_moves)
    for k in range(n):
        idx = indices[k]
        p = priorities[k]
        for t in range(N):
            if p == threshold[t]:
                mask[idx] = 1
                break
    return mask


def _pack_bits_bool2d(arr2d: np.ndarray) -> np.ndarray:
    """
    Pack a 2D 0/1 or bool array into a 1D uint8 bit vector using bitorder='big'.
    """
    # Ensure uint8 0/1
    a = arr2d.astype(np.uint8, copy=False)
    return np.packbits(a.reshape(-1), bitorder='big')


def _unpack_bits_to_2d(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Unpack a 1D uint8 bit vector to a 2D uint8 array (0/1) with given shape.
    """
    flat = np.unpackbits(bits, bitorder='big')
    need = rows * cols
    if flat.size > need:
        flat = flat[:need]
    return flat.reshape((rows, cols)).astype(np.uint8, copy=False)


def _bit_clear_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Clear (set to 0) the bit at flat index idx in the packed array (bitorder='big').
    Uses a non-negative mask to avoid OverflowError from bitwise NOT on Python ints.
    """
    byte_i = idx // 8
    off    = idx % 8
    # Build a clear mask: 0xFF with target bit cleared
    clear_mask = np.uint8(0xFF ^ (1 << (7 - off)))
    bits[byte_i] &= clear_mask


def _bit_set_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Set (to 1) the bit at flat index idx in the packed array (bitorder='big').
    """
    byte_i = idx // 8
    off    = idx % 8
    bits[byte_i] |= np.uint8(1 << (7 - off))


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
