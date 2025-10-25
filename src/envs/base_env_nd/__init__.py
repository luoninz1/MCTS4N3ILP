"""
Base Environment Package

This package provides the abstract base class for n-dimensional environments
compatible with MCTS algorithms.

Main Components:
- base_env_nd.py: Base environment class for MCTS-like algorithms
"""

from src.envs.base_env_nd.base_env_nd import BaseEnvND

__all__ = [
    'BaseEnvND'
]