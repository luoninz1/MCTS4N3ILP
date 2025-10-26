"""Unit tests for reward functions."""
import os
import sys
import numpy as np
import pytest

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.rewards.n3il_rewards import get_value_nb, get_value_exp_norm_nb
from src.algos.mcts import set_seeds


class TestGetValueNb:
    """Tests for get_value_nb reward function."""

    def test_value_range_empty(self):
        """Value for empty state is finite and positive."""
        state = np.zeros((4, 4), dtype=np.uint8)
        value = get_value_nb(state, pts_upper_bound=8)

        assert np.isfinite(value)
        assert value > 0  # Exponential reward is always positive

    def test_value_range_with_points(self):
        """Value with points is finite and positive."""
        state = np.zeros((4, 4), dtype=np.uint8)
        state[0, 0] = 1
        state[1, 1] = 1
        state[2, 2] = 1

        value = get_value_nb(state, pts_upper_bound=8)

        assert np.isfinite(value)
        assert value > 0

    def test_value_deterministic(self):
        """Same state always gives same value."""
        state = np.zeros((5, 5), dtype=np.uint8)
        state[0, 0] = 1
        state[1, 2] = 1
        state[3, 4] = 1

        value1 = get_value_nb(state.copy(), pts_upper_bound=10)
        value2 = get_value_nb(state.copy(), pts_upper_bound=10)

        assert value1 == value2

    def test_value_monotone_more_points(self):
        """More points affects value (reward function uses exponential)."""
        state1 = np.zeros((4, 4), dtype=np.uint8)
        state1[0, 0] = 1

        state2 = state1.copy()
        state2[2, 2] = 1

        value1 = get_value_nb(state1, pts_upper_bound=8)
        value2 = get_value_nb(state2, pts_upper_bound=8)

        # Both values should be valid
        assert np.isfinite(value1) and np.isfinite(value2)
        assert value1 > 0 and value2 > 0
        # Exponential reward: more points changes the value
        assert value1 != value2

    def test_value_upper_bound_effect(self):
        """pts_upper_bound affects the reward calculation."""
        state = np.zeros((4, 4), dtype=np.uint8)
        state[0, 0] = 1
        state[1, 1] = 1

        value1 = get_value_nb(state.copy(), pts_upper_bound=8)
        value2 = get_value_nb(state.copy(), pts_upper_bound=16)

        # Upper bound affects normalization
        assert np.isfinite(value1) and np.isfinite(value2)
        assert value1 != value2


class TestGetValueExpNormNb:
    """Tests for get_value_exp_norm_nb variant."""

    def test_exp_norm_value_range(self):
        """Exponential normalized value is finite and positive."""
        state = np.zeros((4, 4), dtype=np.uint8)
        state[0, 0] = 1
        state[1, 1] = 1

        value = get_value_exp_norm_nb(state, pts_upper_bound=8)

        assert np.isfinite(value)
        assert value > 0

    def test_exp_norm_deterministic(self):
        """Same state always gives same value."""
        state = np.zeros((4, 4), dtype=np.uint8)
        state[0, 0] = 1
        state[2, 3] = 1

        value1 = get_value_exp_norm_nb(state.copy(), pts_upper_bound=8)
        value2 = get_value_exp_norm_nb(state.copy(), pts_upper_bound=8)

        assert value1 == value2

    def test_exp_norm_empty_state(self):
        """Empty state has defined value."""
        state = np.zeros((4, 4), dtype=np.uint8)

        value = get_value_exp_norm_nb(state, pts_upper_bound=8)

        assert np.isfinite(value)
        assert value > 0


class TestRewardConsistency:
    """Cross-function consistency tests."""

    def test_both_functions_in_valid_range(self):
        """Both reward functions return finite positive values."""
        set_seeds(42)
        state = np.zeros((5, 5), dtype=np.uint8)
        # Add random points
        for _ in range(5):
            r, c = np.random.randint(0, 5, size=2)
            state[r, c] = 1

        value_linear = get_value_nb(state, pts_upper_bound=10)
        value_exp = get_value_exp_norm_nb(state, pts_upper_bound=10)

        assert np.isfinite(value_linear) and value_linear > 0
        assert np.isfinite(value_exp) and value_exp > 0
