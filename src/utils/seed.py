"""Random seed management and numba warmup for reproducibility."""
import numpy as np
import random
from numba import config

# Set threading layer once at module import (not per-call)
config.THREADING_LAYER = 'safe'

def set_seeds(seed: int, worker_id: int = 0) -> int:
    """
    Set random seeds for Python, NumPy, and ensure deterministic RNG state.

    Args:
        seed: Base random seed
        worker_id: Worker ID for deterministic per-worker seeding (default 0)

    Returns:
        effective_seed: The actual seed used (seed + worker_id * 10000)
    """
    effective_seed = seed + worker_id * 10000
    np.random.seed(effective_seed)
    random.seed(effective_seed)
    return effective_seed

def warmup_numba():
    """
    Warm up numba functions with seeded state to ensure deterministic JIT.
    Call once after set_seeds() at process startup.
    """
    # Import here to avoid circular dependencies
    from src.envs.collinear_for_mcts import simulate_nb

    # Trigger compilation with a tiny dummy state
    dummy_state = np.zeros((2, 2), dtype=np.uint8)
    _ = simulate_nb(dummy_state, 2, 2, 4)
