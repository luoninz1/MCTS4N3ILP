"""
D4 Symmetry group implementation for No-Three-In-Line.

This module implements the D4 symmetry group for grid-based problems.
The D4 group consists of 8 elements: E (identity), R (90° CCW rotation),
R2 (180° rotation), R3 (270° CCW rotation), SV (vertical reflection),
SH (horizontal reflection), SD (main diagonal reflection),
SA (anti-diagonal reflection).
"""

import numpy as np
from numba import njit

# D4 element codes (fixed enumeration)
E, R, R2, R3, SV, SH, SD, SA = 0, 1, 2, 3, 4, 5, 6, 7


@njit(cache=True, nogil=True)
def _map_coord(i, j, elem, row_count, col_count):
    """
    Map coordinates (i, j) under a D4 element 'elem' on an (row_count x col_count) grid.
    For non-square grids, only {E, R2, SV, SH} are meaningful; we never call others there.
    """
    if elem == E:   # identity
        return i, j
    elif elem == R:  # rotate 90° CCW (only valid for square)
        # (i, j) -> (j, n-1-i)
        return j, col_count - 1 - i
    elif elem == R2:  # rotate 180°
        # (i, j) -> (m-1-i, n-1-j)
        return row_count - 1 - i, col_count - 1 - j
    elif elem == R3:  # rotate 270° CCW (90° CW)
        # (i, j) -> (n-1-j, i)
        return row_count - 1 - j, i
    elif elem == SV:  # reflect vertical axis (left-right flip)
        # (i, j) -> (i, n-1-j)
        return i, col_count - 1 - j
    elif elem == SH:  # reflect horizontal axis (up-down flip)
        # (i, j) -> (m-1-i, j)
        return row_count - 1 - i, j
    elif elem == SD:  # reflect main diagonal y=x (only valid for square)
        # (i, j) -> (j, i)
        return j, i
    elif elem == SA:  # reflect anti-diagonal y=-x (only valid for square)
        # (i, j) -> (n-1-j, m-1-i)
        return col_count - 1 - j, row_count - 1 - i
    else:
        return i, j


@njit(cache=True, nogil=True)
def apply_element_to_action(action, elem, row_count, col_count):
    """Apply a D4 element to a flattened action index."""
    i = action // col_count
    j = action % col_count
    ni, nj = _map_coord(i, j, elem, row_count, col_count)
    return ni * col_count + nj


@njit(cache=True, nogil=True)
def _element_fixes_state(elem, state):
    """
    Check if D4 element 'elem' fixes 'state' pointwise.
    We compare state[i,j] with state[ mapped(i,j) ] for all cells.
    """
    m, n = state.shape
    for i in range(m):
        for j in range(n):
            ii, jj = _map_coord(i, j, elem, m, n)
            if state[i, j] != state[ii, jj]:
                return False
    return True


@njit(cache=True, nogil=True)
def detect_stabilizer_elements_nb(state):
    """
    Return an 8-length boolean array 'fix' where fix[elem]=True
    iff the D4 element 'elem' fixes the state.

    For non-square grids, we skip checks for R, R3, SD, SA (set to False).
    """
    m, n = state.shape
    square = (m == n)

    fix = np.zeros(8, dtype=np.bool_)
    # Always consider E, R2, SV, SH
    fix[E]  = _element_fixes_state(E,  state)
    fix[R2] = _element_fixes_state(R2, state)
    fix[SV] = _element_fixes_state(SV, state)
    fix[SH] = _element_fixes_state(SH, state)

    if square:
        fix[R]  = _element_fixes_state(R,  state)
        fix[R3] = _element_fixes_state(R3, state)
        fix[SD] = _element_fixes_state(SD, state)
        fix[SA] = _element_fixes_state(SA, state)
    else:
        fix[R] = False
        fix[R3] = False
        fix[SD] = False
        fix[SA] = False

    return fix


@njit(cache=True, nogil=True)
def _fill_row(row, elems):
    """
    Helper: write a subgroup's element indices into a row (length 8),
    fill unused slots with -1.
    """
    for k in range(8):
        row[k] = -1
    for k in range(len(elems)):
        row[k] = elems[k]


@njit(cache=True, nogil=True)
def _build_canonical_subgroups():
    """
    Returns:
      subs (10 x 8 int array): each row lists the element indices of a canonical subgroup, -1 padded
      sizes (10,): number of elements in each subgroup row
      ids (10,): arbitrary IDs 0..9 for reference
        0:{e}, 1:<r>, 2:<r^2>, 3:<s_v>, 4:<s_h>, 5:<s_d>, 6:<s_a>, 7:V1, 8:V2, 9:D4
    """
    subs = np.empty((10, 8), dtype=np.int64)
    sizes = np.empty(10, dtype=np.int64)
    ids = np.arange(10, dtype=np.int64)

    # 0: {e}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E]))
    subs[0] = row; sizes[0] = 1

    # 1: <r> = {e, r, r2, r3}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, R, R2, R3]))
    subs[1] = row; sizes[1] = 4

    # 2: <r^2> = {e, r2}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, R2]))
    subs[2] = row; sizes[2] = 2

    # 3: <s_v> = {e, s_v}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, SV]))
    subs[3] = row; sizes[3] = 2

    # 4: <s_h> = {e, s_h}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, SH]))
    subs[4] = row; sizes[4] = 2

    # 5: <s_d> = {e, s_d}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, SD]))
    subs[5] = row; sizes[5] = 2

    # 6: <s_a> = {e, s_a}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, SA]))
    subs[6] = row; sizes[6] = 2

    # 7: V1 = {e, r2, s_v, s_h}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, R2, SV, SH]))
    subs[7] = row; sizes[7] = 4

    # 8: V2 = {e, r2, s_d, s_a}
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, R2, SD, SA]))
    subs[8] = row; sizes[8] = 4

    # 9: D4 (all eight)
    row = np.empty(8, dtype=np.int64); _fill_row(row, np.array([E, R, R2, R3, SV, SH, SD, SA]))
    subs[9] = row; sizes[9] = 8

    return subs, sizes, ids


@njit(cache=True, nogil=True)
def _fixes_equals_subgroup(fix, subs_row):
    """
    Check whether the boolean 'fix' set equals the subgroup listed in 'subs_row'.
    """
    listed = np.zeros(8, dtype=np.bool_)
    for k in range(8):
        idx = subs_row[k]
        if idx == -1:
            break
        listed[idx] = True

    # exact equality
    for e in range(8):
        if fix[e] != listed[e]:
            return False
    return True


@njit(cache=True, nogil=True)
def identify_stabilizer_subgroup_nb(state):
    """
    Detect the stabilizer elements, then match exactly to one of the 10 canonical subgroups.
    Returns:
      subgroup_id (0..9 as documented above),
      subgroup_elems (length <= 8, filled with -1 beyond size),
      subgroup_size
    If no exact match (shouldn't happen), fall back to the literal 'fix' set.
    """
    fix = detect_stabilizer_elements_nb(state)
    subs, sizes, ids = _build_canonical_subgroups()

    # Try to match exactly one canonical subgroup
    for r in range(10):
        if _fixes_equals_subgroup(fix, subs[r]):
            return ids[r], subs[r], sizes[r]

    # Fallback: construct subgroup row directly from 'fix'
    # (This would be unusual; included for robustness.)
    tmp = np.empty(8, dtype=np.int64)
    cnt = 0
    for e in range(8):
        if fix[e]:
            tmp[cnt] = e
            cnt += 1
    for k in range(cnt, 8):
        tmp[k] = -1
    return -1, tmp, cnt  # id -1 = non-canonical (should not occur)


@njit(cache=True, nogil=True)
def filter_actions_by_stabilizer_nb(valid_moves, state, row_count, col_count):
    """
    Reduce action space by orbits under the stabilizer subgroup G_x of the current state.
    Keep the minimum flattened index in each orbit.

    Args:
      valid_moves: 1D boolean array
      state: 2D uint8 array
      row_count, col_count: ints

    Returns:
      filtered_moves: 1D boolean array
    """
    # Identify the stabilizer subgroup (one of the 10)
    subgroup_id, subgroup_row, subgroup_size = identify_stabilizer_subgroup_nb(state)

    # If stabilizer is trivial {e}, return original
    if subgroup_size <= 1:
        return valid_moves

    filtered = valid_moves.copy()
    N = valid_moves.shape[0]

    # Pre-extract subgroup elements into a compact array
    elems = np.empty(subgroup_size, dtype=np.int64)
    for k in range(subgroup_size):
        elems[k] = subgroup_row[k]  # no -1 within subgroup_size

    # Iterate over valid indices; for each orbit, keep the minimal index
    idxs = np.where(valid_moves)[0]
    for t in range(idxs.shape[0]):
        a = idxs[t]
        if not filtered[a]:
            continue

        # Build orbit under G_x
        min_a = a
        orbit = np.empty(subgroup_size, dtype=np.int64)
        for k in range(subgroup_size):
            b = apply_element_to_action(a, elems[k], row_count, col_count)
            orbit[k] = b
            if b < min_a:
                min_a = b

        # Disable non-canonical members (keep only min_a)
        for k in range(subgroup_size):
            b = orbit[k]
            if b != min_a and b < N:
                filtered[b] = False

    return filtered
