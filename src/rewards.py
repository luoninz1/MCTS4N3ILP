"""
Centralized reward/value functions for different game environments.

This module provides a registry of value functions that can be selected
via configuration. This allows easy experimentation with different reward
schemes without modifying code.
"""

import numpy as np
from numba import njit
from typing import Dict, Callable


# Registry of available value functions
VALUE_FUNCTION_REGISTRY: Dict[str, Callable] = {}


def register_value_function(name: str):
    """
    Decorator to register a value function.

    Usage:
        @register_value_function('my_value_fn')
        @njit
        def my_value_fn(state, pts_upper_bound):
            return score / pts_upper_bound
    """
    def decorator(func):
        VALUE_FUNCTION_REGISTRY[name] = func
        return func
    return decorator


@njit(cache=True, nogil=True)
@register_value_function('point_count')
def point_count_value(state, pts_upper_bound):
    """
    Default value function: normalized point count.

    Returns the number of points placed, normalized by the upper bound.
    This is the standard reward for point-placement games like N3ILP.

    Args:
        state: 2D numpy array representing the grid state
        pts_upper_bound: Maximum expected number of points

    Returns:
        float: Normalized value between 0 and 1
    """
    num_points = np.sum(state)
    return num_points / pts_upper_bound


@njit(cache=True, nogil=True)
@register_value_function('point_count_squared')
def point_count_squared_value(state, pts_upper_bound):
    """
    Quadratic reward: encourages maximizing points more aggressively.

    Returns the square of normalized point count.
    This gives higher reward for larger point counts.

    Args:
        state: 2D numpy array representing the grid state
        pts_upper_bound: Maximum expected number of points

    Returns:
        float: Normalized squared value between 0 and 1
    """
    num_points = np.sum(state)
    normalized = num_points / pts_upper_bound
    return normalized * normalized


@njit(cache=True, nogil=True)
@register_value_function('point_count_log')
def point_count_log_value(state, pts_upper_bound):
    """
    Logarithmic reward: less aggressive scaling.

    Uses log(1 + points) to provide diminishing returns for higher counts.

    Args:
        state: 2D numpy array representing the grid state
        pts_upper_bound: Maximum expected number of points

    Returns:
        float: Normalized log value
    """
    num_points = np.sum(state)
    max_log = np.log(1 + pts_upper_bound)
    return np.log(1 + num_points) / max_log


def get_value_function(name: str = 'point_count') -> Callable:
    """
    Get a value function by name.

    Args:
        name: Name of the value function (default: 'point_count')

    Returns:
        Callable value function

    Raises:
        ValueError: If value function name is not registered

    Example:
        value_fn = get_value_function('point_count')
        score = value_fn(state, upper_bound)
    """
    if name not in VALUE_FUNCTION_REGISTRY:
        available = ', '.join(VALUE_FUNCTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown value function: {name}. "
            f"Available functions: {available}"
        )
    return VALUE_FUNCTION_REGISTRY[name]


def list_value_functions():
    """List all registered value functions."""
    return list(VALUE_FUNCTION_REGISTRY.keys())


# Backward compatibility: export the default function with old name
get_value_nb = point_count_value
