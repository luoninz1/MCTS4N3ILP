"""
No-Three-In-Line environment base class.

This module provides the core environment implementation for the
No-Three-In-Line problem, compatible with MCTS algorithms.
"""

import numpy as np
import random
import datetime
import os
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray
import numpy as np

# This is not gym-compatible, but serves as a base class for MCTS-like algorithms
class BaseEnvND:
    def __init__(self, shape: tuple, args: dict):
        """
        Shape can be a tuple of any dimension, e.g., (6, 7) for 2D grid."""
        self.shape = shape
        self.dimensions = len(shape)

    def get_initial_state(self) -> NDArray[np.uint8]:
        """Returns the initial state as a multi-dimensional array of zeros."""
        return np.zeros(self.shape, np.uint8)

    def get_next_state(self, state: NDArray[np.uint8], action: tuple) -> NDArray[np.uint8]:
        """Takes the current state and an action, returns the next state. Note the original state is modified."""
        state[action] = 1
        return state
    
    def get_action_space(self, state: NDArray[np.uint8], action: tuple) -> NDArray[np.uint8]:
        """Must be overridden by subclass"""
        raise NotImplementedError("Subclasses must implement get_action_space.")
    
    def get_action_space_subset(self, parent_state: NDArray[np.uint8], parent_valid_moves: NDArray[np.uint8], action_taken: tuple) -> NDArray[np.uint8]:
        """Must be overridden by subclass (If applicable)"""
        raise NotImplementedError("Subclasses must implement get_valid_moves_subset (If applicable).")
    
    def get_terminal_state_by_simulation(self, state: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Must be overridden by subclass.
        Input a non-terminal state, simulate until terminal state is reached, and return that terminal state.
        """
        raise NotImplementedError("Subclasses must implement get_terminal_state_by_simulation.")
    
    def get_value_and_terminated(self, state:  NDArray[np.uint8], action_space: NDArray[np.uint8]) -> Tuple[float, bool]:
        """
        Return the normalized value and terminal status of the current state.
        Delegates value calculation to get_value_nb().
        """
        raise NotImplementedError("Subclasses must implement get_value_and_terminated.")
