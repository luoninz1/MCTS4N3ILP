"""Integration smoke tests for MCTS."""
import os
import sys
import numpy as np
import pytest

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.algos.mcts import evaluate, MCTS
from src.utils.seed import set_seeds, warmup_numba
from src.envs import N3il, N3il_with_symmetry


class TestMCTSSingleRun:
    """Integration test for single MCTS run."""

    def test_mcts_single_run_deterministic(self):
        """Fixed seed produces reproducible results."""
        args = {
            'n': 4,
            'num_searches': 50,
            'C': 1.41,
            'environment': 'N3il_with_symmetry',
            'algorithm': 'MCTS',
            'max_level_to_use_symmetry': 10,
            'node_compression': True,
            'virtual_loss': 1.0,
            'random_seed': 42,
            'display_state': False,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'logging_mode': True,
            'TopN': 4,
            'simulate_with_priority': False,
        }

        # Run 1
        set_seeds(42)
        warmup_numba()
        result1 = evaluate(args.copy())

        # Run 2 with same seed
        set_seeds(42)
        warmup_numba()
        result2 = evaluate(args.copy())

        # Should produce identical results
        assert result1 == result2
        assert result1 > 0  # Should place at least one point

    def test_mcts_completes_without_error(self):
        """MCTS runs to completion without crashing."""
        args = {
            'n': 3,
            'num_searches': 20,
            'C': 1.41,
            'environment': 'N3il',
            'algorithm': 'MCTS',
            'max_level_to_use_symmetry': 0,
            'node_compression': False,
            'virtual_loss': 1.0,
            'random_seed': 123,
            'display_state': False,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'logging_mode': True,
            'TopN': 3,
            'simulate_with_priority': False,
        }

        result = evaluate(args)

        assert result is not None
        assert result >= 0

    def test_mcts_search_returns_valid_probs(self):
        """MCTS.search returns valid probability distribution."""
        set_seeds(42)
        args = {
            'n': 4,
            'num_searches': 30,
            'C': 1.41,
            'max_level_to_use_symmetry': 0,
            'node_compression': False,
            'virtual_loss': 1.0,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'TopN': 4,
            'simulate_with_priority': False,
        }

        env = N3il(grid_size=(4, 4), args=args, priority_grid=None)
        mcts = MCTS(env, args=args)
        state = env.get_initial_state()

        probs = mcts.search(state)

        assert probs.shape == (16,)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_mcts_with_symmetry_completes(self):
        """MCTS with symmetry filtering completes successfully."""
        set_seeds(42)
        args = {
            'n': 4,
            'num_searches': 30,
            'C': 1.41,
            'max_level_to_use_symmetry': 10,
            'node_compression': False,
            'virtual_loss': 1.0,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'TopN': 4,
            'simulate_with_priority': False,
        }

        env = N3il_with_symmetry(grid_size=(4, 4), args=args, priority_grid=None)
        mcts = MCTS(env, args=args)
        state = env.get_initial_state()

        probs = mcts.search(state)

        assert probs.shape == (16,)
        assert np.isclose(np.sum(probs), 1.0)


class TestEvaluateFunction:
    """Tests for the evaluate() entry point."""

    def test_evaluate_returns_valid_result(self):
        """evaluate() returns valid terminal point count."""
        args = {
            'n': 3,
            'num_searches': 20,
            'C': 1.41,
            'environment': 'N3il',
            'algorithm': 'MCTS',
            'max_level_to_use_symmetry': 0,
            'node_compression': False,
            'virtual_loss': 1.0,
            'random_seed': 999,
            'display_state': False,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'logging_mode': True,
            'TopN': 3,
            'simulate_with_priority': False,
        }

        result = evaluate(args)

        assert isinstance(result, (int, np.integer))
        assert result > 0

    def test_evaluate_with_different_envs(self):
        """evaluate() works with different environments."""
        base_args = {
            'n': 3,
            'num_searches': 15,
            'C': 1.41,
            'algorithm': 'MCTS',
            'max_level_to_use_symmetry': 5,
            'node_compression': False,
            'virtual_loss': 1.0,
            'random_seed': 111,
            'display_state': False,
            'tree_visualization': False,
            'pause_at_each_step': False,
            'process_bar': False,
            'logging_mode': True,
            'TopN': 3,
            'simulate_with_priority': False,
        }

        # Test N3il
        args1 = {**base_args, 'environment': 'N3il'}
        result1 = evaluate(args1)
        assert result1 > 0

        # Test N3il_with_symmetry
        args2 = {**base_args, 'environment': 'N3il_with_symmetry'}
        result2 = evaluate(args2)
        assert result2 > 0
