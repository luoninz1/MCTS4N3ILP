"""
Reward/value functions for No-Three-In-Line problem.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def get_value_nb(state, pts_upper_bound):
    """
    Calculate normalized value for a terminal state.
    Returns the number of points placed, normalized by the upper bound.

    Args:
        state: 2D numpy array representing the grid state
        pts_upper_bound: Maximum expected number of points

    Returns:
        float: Normalized value between 0 and 1
    """
    num_points = np.sum(state)
    return num_points / pts_upper_bound
