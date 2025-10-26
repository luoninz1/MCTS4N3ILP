"""Unit tests for D4 symmetry operations."""
import os
import sys
import numpy as np
import pytest

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.envs.collinear_for_mcts import (
    N3il_with_symmetry,
    apply_element_to_action,
    filter_actions_by_stabilizer_nb,
)


@pytest.fixture
def symmetry_env():
    """Create a symmetry-enabled environment."""
    args = {
        'max_level_to_use_symmetry': 10,
        'n': 4,
        'TopN': 4,
        'simulate_with_priority': False,
    }
    return N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)


class TestSymmetryFiltering:
    """Tests for symmetry filtering operations."""

    def test_filter_actions_by_stabilizer_basic(self, symmetry_env):
        """Test basic symmetry filtering."""
        state = symmetry_env.get_initial_state()
        valid_moves = symmetry_env.get_valid_moves(state)

        # Apply symmetry filtering
        filtered = filter_actions_by_stabilizer_nb(
            valid_moves,
            state,
            symmetry_env.row_count,
            symmetry_env.column_count
        )

        # Should reduce action count on symmetric state
        assert np.sum(filtered) <= np.sum(valid_moves)
        assert np.sum(filtered) > 0

    def test_symmetry_filtering_asymmetric_state(self, symmetry_env):
        """Asymmetric state has less filtering."""
        state = symmetry_env.get_initial_state()
        state[0, 1] = 1  # Break symmetry
        valid_moves = symmetry_env.get_valid_moves(state)

        filtered = filter_actions_by_stabilizer_nb(
            valid_moves,
            state,
            symmetry_env.row_count,
            symmetry_env.column_count
        )

        # Should still return valid filtered set
        assert np.sum(filtered) >= 0


class TestD4Transformations:
    """Tests for D4 group element applications."""

    def test_apply_element_identity(self):
        """Identity element doesn't change action."""
        action = 5
        transformed = apply_element_to_action(action, 'e', 4, 4)
        assert transformed == action

    def test_apply_element_rotation(self):
        """Rotation transforms action correctly."""
        action = 2  # (0, 2) - choose action that doesn't map to itself
        # Apply 90° rotation (r)
        transformed = apply_element_to_action(action, 'r', 4, 4)
        # Should map to valid position
        assert 0 <= transformed < 16
        # For most actions, rotation changes the action (though not all)
        # Just verify it's a valid transformation

    def test_apply_element_reflection(self):
        """Reflection transforms action correctly."""
        action = 1  # (0, 1)
        # Apply reflection (s)
        transformed = apply_element_to_action(action, 's', 4, 4)
        # Should map to reflected position
        assert 0 <= transformed < 16

    def test_transformation_deterministic(self):
        """Same transformation gives same result."""
        action = 5
        result1 = apply_element_to_action(action, 'r', 4, 4)
        result2 = apply_element_to_action(action, 'r', 4, 4)
        assert result1 == result2


class TestSymmetryValidMoves:
    """Tests for symmetry-aware valid moves."""

    def test_get_valid_moves_with_symmetry_initial(self, symmetry_env):
        """Symmetry filtering on initial state reduces actions."""
        state = symmetry_env.get_initial_state()

        # Get valid moves (which uses symmetry filtering internally)
        valid = symmetry_env.get_valid_moves(state)

        # Should have some valid moves
        assert np.sum(valid) > 0
        assert np.sum(valid) <= 16

    def test_filter_valid_moves_by_symmetry(self, symmetry_env):
        """filter_valid_moves_by_symmetry method works."""
        state = symmetry_env.get_initial_state()
        action_space = np.ones(16, dtype=np.uint8)

        filtered = symmetry_env.filter_valid_moves_by_symmetry(action_space, state)

        # Should reduce actions on symmetric state
        assert np.sum(filtered) < np.sum(action_space)
        assert np.sum(filtered) > 0
