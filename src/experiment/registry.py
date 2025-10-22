"""
Registry for algorithms and environments.

This module provides a centralized registry for dynamically loading
algorithms and environments based on configuration strings.
"""

from typing import Dict, Type, Any


# Algorithm registry - maps algorithm names to their classes
ALGORITHM_REGISTRY: Dict[str, Type] = {}

# Environment registry - maps environment names to their classes
ENVIRONMENT_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str):
    """
    Decorator to register an algorithm class.

    Usage:
        @register_algorithm('MCTS')
        class MCTS:
            ...
    """
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def register_environment(name: str):
    """
    Decorator to register an environment class.

    Usage:
        @register_environment('N3il')
        class N3il:
            ...
    """
    def decorator(cls):
        ENVIRONMENT_REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm_class(name: str) -> Type:
    """
    Get algorithm class by name.

    Args:
        name: Algorithm name (e.g., 'MCTS', 'MCGS')

    Returns:
        Algorithm class

    Raises:
        ValueError: If algorithm name is not registered
    """
    _populate_registries()  # Lazy initialization
    if name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm: {name}. "
            f"Available algorithms: {available}"
        )
    return ALGORITHM_REGISTRY[name]


def get_environment_class(name: str) -> Type:
    """
    Get environment class by name.

    Args:
        name: Environment name (e.g., 'N3il', 'N3il_with_symmetry')

    Returns:
        Environment class

    Raises:
        ValueError: If environment name is not registered
    """
    _populate_registries()  # Lazy initialization
    if name not in ENVIRONMENT_REGISTRY:
        available = ', '.join(ENVIRONMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment: {name}. "
            f"Available environments: {available}"
        )
    return ENVIRONMENT_REGISTRY[name]


def _populate_registries():
    """
    Populate registries with available algorithms and environments.
    Called lazily on first use to avoid circular imports.
    """
    if ALGORITHM_REGISTRY or ENVIRONMENT_REGISTRY:
        return  # Already populated

    # Import and register algorithms
    try:
        from src.algos.mcts.tree_search import MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS
        ALGORITHM_REGISTRY['MCTS'] = MCTS
        ALGORITHM_REGISTRY['ParallelMCTS'] = ParallelMCTS
        ALGORITHM_REGISTRY['LeafChildParallelMCTS'] = LeafChildParallelMCTS
        ALGORITHM_REGISTRY['MCGS'] = MCGS
    except ImportError as e:
        print(f"Warning: Could not import MCTS algorithms: {e}")

    # Import and register environments
    try:
        from src.envs.n3il.n3il_env import N3il
        from src.envs.n3il.n3il_symmetry_env import N3il_with_symmetry
        ENVIRONMENT_REGISTRY['N3il'] = N3il
        ENVIRONMENT_REGISTRY['N3il_with_symmetry'] = N3il_with_symmetry
    except ImportError as e:
        print(f"Warning: Could not import environments: {e}")


# Don't populate on module import - do it lazily to avoid circular imports


def list_algorithms():
    """List all registered algorithms."""
    _populate_registries()  # Lazy initialization
    return list(ALGORITHM_REGISTRY.keys())


def list_environments():
    """List all registered environments."""
    _populate_registries()  # Lazy initialization
    return list(ENVIRONMENT_REGISTRY.keys())
