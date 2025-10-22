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


def get_mcts_with_symmetry_config(n, random_seed=0, num_searches_multiplier=100, max_symmetry_level=1):
    """
    Get MCTS configuration with symmetry enabled.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        max_symmetry_level (int): Maximum level to use symmetry (0 = disabled)

    Returns:
        dict: MCTS with symmetry configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'environment': 'N3il_with_symmetry',
        'max_level_to_use_symmetry': max_symmetry_level,
    })
    return config


def get_mcts_compressed_config(n, random_seed=0, num_searches_multiplier=100):
    """
    Get MCTS configuration with node compression enabled.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches

    Returns:
        dict: MCTS with node compression configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'node_compression': True,
    })
    return config


def get_parallel_mcts_config(n, random_seed=0, num_searches_multiplier=100, num_workers=4):
    """
    Get parallel MCTS configuration.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        num_workers (int): Number of parallel workers

    Returns:
        dict: Parallel MCTS configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'num_workers': num_workers,
    })
    return config


def get_mcts_with_priority_config(n, random_seed=0, num_searches_multiplier=100, top_n=None):
    """
    Get MCTS configuration with priority-based simulation.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        top_n (int): Number of top priority moves to consider (default: n)

    Returns:
        dict: MCTS with priority configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'simulate_with_priority': True,
        'TopN': top_n if top_n is not None else n,
    })
    return config


def get_leaf_child_parallel_config(n, random_seed=0, num_searches_multiplier=100,
                                   num_workers=4, simulations_per_leaf=1, child_parallel=True):
    """
    Get Leaf/Child parallel MCTS configuration.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        num_workers (int): Number of parallel workers
        simulations_per_leaf (int): Number of simulations per leaf
        child_parallel (bool): Enable child-parallel expansion

    Returns:
        dict: Leaf/Child parallel MCTS configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'algorithm': 'LeafChildParallelMCTS',
        'num_workers': num_workers,
        'simulations_per_leaf': simulations_per_leaf,
        'child_parallel': child_parallel,
    })
    return config


def get_mcgs_config(n, random_seed=0, num_searches_multiplier=100, gamma=0.99):
    """
    Get Monte Carlo Graph Search (MCGS) configuration.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        gamma (float): Discount factor for MCGS

    Returns:
        dict: MCGS configuration parameters
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'algorithm': 'MCGS',
        'gamma': gamma,
    })
    return config


def get_mcts_with_custom_reward_config(n, random_seed=0, num_searches_multiplier=100, value_function='point_count'):
    """
    Get MCTS configuration with custom reward/value function.

    Args:
        n (int): Grid size
        random_seed (int): Random seed for reproducibility
        num_searches_multiplier (int): Multiplier for number of searches
        value_function (str): Name of value function to use
            Options: 'point_count', 'point_count_squared', 'point_count_log'

    Returns:
        dict: MCTS configuration with custom value function

    Example:
        # Use quadratic reward (encourages higher scores)
        config = get_mcts_with_custom_reward_config(10, value_function='point_count_squared')
    """
    config = get_mcts_config(n, random_seed, num_searches_multiplier)
    config.update({
        'value_function': value_function,
    })
    return config


def set_output_directories(config, test_name):
    """
    Set table and figure output directories for a configuration.

    Args:
        config (dict): Configuration dictionary
        test_name (str): Name of the test (used for directory structure)

    Returns:
        dict: Updated configuration with output directories
    """
    base_dir = os.path.dirname(__file__)
    config['table_dir'] = os.path.join(base_dir, 'results', test_name)
    config['figure_dir'] = os.path.join(base_dir, 'figures', test_name)
    return config


# Preset configurations for common experiment types
PRESETS = {
    'small_grid': lambda n: get_mcts_config(n, num_searches_multiplier=50),
    'medium_grid': lambda n: get_mcts_config(n, num_searches_multiplier=100),
    'large_grid': lambda n: get_mcts_config(n, num_searches_multiplier=200),
    'quick_test': lambda n: get_mcts_config(n, num_searches_multiplier=10),
    'symmetry_test': lambda n: get_mcts_with_symmetry_config(n, max_symmetry_level=2),
    'parallel_test': lambda n: get_parallel_mcts_config(n, num_workers=4),
}


def get_preset_config(preset_name, n, random_seed=0):
    """
    Get a preset configuration by name.

    Args:
        preset_name (str): Name of the preset
        n (int): Grid size
        random_seed (int): Random seed for reproducibility

    Returns:
        dict: Preset configuration parameters

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESETS.keys())}")

    config = PRESETS[preset_name](n)
    config['random_seed'] = random_seed
    return config
