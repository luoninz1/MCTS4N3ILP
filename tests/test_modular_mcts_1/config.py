"""
Configuration module for MCTS experiments.

This module provides configuration presets and parameter management
for No-Three-In-Line MCTS experiments.
"""

import os


def get_base_config():
    """Get base configuration common to all experiments."""
    return {
        'process_bar': True,
        'display_state': True,
        'logging_mode': True,
        'table_dir': None,  # Will be set per test
        'figure_dir': None,  # Will be set per test
        'tree_visualization': False,
        'pause_at_each_step': False,
        'value_function': 'point_count',  # Default reward function
    }


def get_mcts_config(n, random_seed=0, num_searches_multiplier=100):
    """
    Get MCTS-specific configuration.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches (searches = multiplier * n^2)

    Returns:
        dict: MCTS configuration parameters
    """
    config = get_base_config()
    config.update({
        'environment': 'N3il',
        'algorithm': 'MCTS',
        'n': n,
        'C': 1.41,
        'num_searches': num_searches_multiplier * (n ** 2),
        'num_workers': 1,
        'virtual_loss': 1.0,
        'TopN': n,
        'simulate_with_priority': False,
        'random_seed': random_seed,
        'node_compression': False,
        'max_level_to_use_symmetry': 0,
    })
    return config