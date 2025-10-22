"""
Reward/value functions for No-Three-In-Line problem.

DEPRECATED: This module is kept for backward compatibility only.
Please use src.rewards instead.
"""

# Import from centralized rewards module
from src.rewards import point_count_value as get_value_nb

__all__ = ['get_value_nb']
