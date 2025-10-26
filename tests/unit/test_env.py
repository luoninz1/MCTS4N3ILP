"""Unit tests for environment API."""
import os
import sys
import numpy as np
import pytest

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.envs import N3il, N3il_with_symmetry
from src.algos.mcts import set_seeds


@pytest.fixture
def basic_args():
    """Basic args dict for environment initialization."""
    return {
        'max_level_to_use_symmetry': 0,
        'n': 4,
        'TopN': 4,  # Required for get_valid_moves with priority_grid
        'simulate_with_priority': False,  # Required for Node.simulate()
    }


class TestN3ilBasics:
    """Tests for N3il environment basic API."""

    def test_get_initial_state(self, basic_args):
        """Initial state returns zeros of correct shape."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()

        assert state.shape == (4, 4)
        assert state.dtype == np.uint8
        assert np.sum(state) == 0

    def test_get_next_state_immutability(self, basic_args):
        """get_next_state DOES mutate input (current behavior to document)."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        original_id = id(state)

        new_state = env.get_next_state(state, 5)

        # Current behavior: get_next_state mutates AND returns the same array
        assert id(new_state) == original_id
        assert np.sum(new_state) == 1
        assert new_state[1, 1] == 1

    def test_get_next_state_action_applied(self, basic_args):
        """get_next_state correctly applies action."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        action = 5  # row=1, col=1 for 4x4

        new_state = env.get_next_state(state, action)

        assert new_state[1, 1] == 1
        assert np.sum(new_state) == 1

    def test_get_valid_moves_initial(self, basic_args):
        """Initial state has all moves valid."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        valid = env.get_valid_moves(state)

        assert valid.shape == (16,)
        assert valid.dtype == np.uint8
        assert np.sum(valid) == 16

    def test_get_valid_moves_after_action(self, basic_args):
        """After action, occupied cell is invalid."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        action = 5

        new_state = env.get_next_state(state, action)
        valid = env.get_valid_moves(new_state)

        assert valid[action] == 0
        assert np.sum(valid) < 16

    def test_get_valid_moves_subset_parity(self, basic_args):
        """Subset matches full recomputation."""
        set_seeds(42)
        env = N3il(grid_size=(5, 5), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        state[0, 0] = 1
        state[2, 2] = 1

        parent_valid = env.get_valid_moves(state)
        valid_actions = np.where(parent_valid)[0]

        if len(valid_actions) > 0:
            action = valid_actions[0]

            # Compute via subset
            subset_valid = env.get_valid_moves_subset(state, parent_valid, action)

            # Compute via full recomputation
            child_state = env.get_next_state(state, action)
            full_valid = env.get_valid_moves(child_state)

            assert np.array_equal(subset_valid, full_valid)

    def test_get_value_and_terminated_initial(self, basic_args):
        """Initial state is not terminal, value in expected range."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        valid = env.get_valid_moves(state)

        value, is_terminal = env.get_value_and_terminated(state, valid)

        assert not is_terminal
        assert 0.0 <= value <= 1.0

    def test_get_value_and_terminated_no_valid_moves(self, basic_args):
        """State with no valid moves is terminal."""
        env = N3il(grid_size=(3, 3), args=basic_args, priority_grid=None)
        state = np.zeros((3, 3), dtype=np.uint8)
        # Create a state where no more moves are valid
        # This is tricky, so we'll mock by passing empty valid_moves
        valid = np.zeros(9, dtype=np.uint8)

        value, is_terminal = env.get_value_and_terminated(state, valid)

        assert is_terminal

    def test_state_to_key_consistency(self, basic_args):
        """state_to_key is hashable and consistent."""
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)
        state = env.get_initial_state()
        state[1, 1] = 1
        state[2, 2] = 1

        key1 = env.state_to_key(state)
        key2 = env.state_to_key(state)

        assert key1 == key2
        assert isinstance(key1, tuple)


class TestN3ilDuplicateInitBug:
    """Test to verify the duplicate init bug (to be fixed in Stage 1)."""

    def test_duplicate_init_no_crash(self, basic_args):
        """Verify duplicate init is fixed and values are correct."""
        # After fixing the duplicate init bug, pts_upper_bound should be row_count * column_count
        env = N3il(grid_size=(4, 4), args=basic_args, priority_grid=None)

        assert env.row_count == 4
        assert env.column_count == 4
        assert env.action_size == 16
        assert env.pts_upper_bound == 16  # Fixed: now using row_count * column_count (not np.min * 2)


class TestN3ilWithSymmetry:
    """Tests for N3il_with_symmetry environment."""

    def test_initialization(self):
        """Verify N3il_with_symmetry initializes without error."""
        args = {
            'max_level_to_use_symmetry': 10,
            'n': 4,
        }
        env = N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)

        assert env.row_count == 4
        assert env.column_count == 4
        assert env.use_symmetry == True
        assert env.max_level_to_use_symmetry == 10

    def test_get_initial_state(self):
        """Initial state works with symmetry environment."""
        args = {
            'max_level_to_use_symmetry': 10,
            'n': 4,
        }
        env = N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)
        state = env.get_initial_state()

        assert state.shape == (4, 4)
        assert np.sum(state) == 0

    def test_filter_valid_moves_by_symmetry(self):
        """Symmetry filtering reduces action space on symmetric states."""
        args = {
            'max_level_to_use_symmetry': 10,
            'n': 4,
            'TopN': 4,  # Required for get_valid_moves when priority_grid is used
        }
        env = N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)
        state = env.get_initial_state()
        action_space = env.get_valid_moves(state)

        filtered = env.filter_valid_moves_by_symmetry(action_space, state)

        # On empty symmetric state, should reduce significantly
        assert np.sum(filtered) < np.sum(action_space)
        assert np.sum(filtered) > 0

    def test_double_super_init_no_crash(self):
        """Verify double super().__init__() doesn't cause immediate crash."""
        # This test documents the current bug (line 1438)
        args = {
            'max_level_to_use_symmetry': 5,
            'n': 4,
        }
        env = N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)

        # Should still work despite duplicate init
        assert env.row_count == 4
        assert env.column_count == 4
