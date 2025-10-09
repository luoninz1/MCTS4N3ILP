"""
Priority functions for the No-Three-In-Line problem.

This module provides priority calculation functions for the grid cells.
Currently, only the supnorm priority is actively used in the experiments.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def supnorm_priority(x: int, y: int) -> float:
    """
    Compute the negative sup-norm priority for a single point.
    Equivalent to: -abs(max(x, y))
    """
    m = x if x > y else y
    return abs(m)


@njit(cache=True)
def supnorm_priority_array(n: int) -> np.ndarray:
    """
    Generate an n-by-n array of priorities using the supnorm_priority function.
    """
    arr = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            arr[i, j] = supnorm_priority(i, j)
    return arr
