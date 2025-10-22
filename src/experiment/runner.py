"""
Experiment runner for algorithm evaluation.

This module provides a flexible experiment runner that can work with
different algorithms and environments based on configuration.
"""

import numpy as np
import time
from typing import Dict, Any, Optional

from src.experiment.registry import get_algorithm_class, get_environment_class


class ExperimentRunner:
    """
    Experiment runner that orchestrates algorithm-environment interactions.

    This class provides a clean interface for running experiments with
    configurable algorithms and environments.

    Example:
        config = {
            'algorithm': 'MCTS',
            'environment': 'N3il',
            'n': 10,
            'num_searches': 1000,
            ...
        }
        runner = ExperimentRunner(config)
        num_points = runner.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary containing:
                - algorithm: Algorithm name (e.g., 'MCTS', 'MCGS')
                - environment: Environment name (e.g., 'N3il', 'N3il_with_symmetry')
                - n: Grid size
                - Other algorithm/environment specific parameters
        """
        self.config = config
        self.env = None
        self.algorithm = None

    def setup(self):
        """Set up the environment and algorithm based on configuration."""
        # Import utilities here to avoid circular imports
        from src.envs import supnorm_priority_array
        from src.algos.mcts.utils import set_seeds
        from src.algos.mcts.simulation import simulate_nb

        # Set random seeds for reproducibility
        if 'random_seed' in self.config:
            set_seeds(self.config['random_seed'])
            # Warmup numba functions with seeded state
            dummy_state = np.zeros((2, 2), dtype=np.int8)
            _ = simulate_nb(dummy_state, 2, 2, 4)

        # Create priority grid
        priority_grid_arr = supnorm_priority_array(self.config['n'])

        # Get environment class and instantiate
        env_class = get_environment_class(self.config['environment'])
        self.env = env_class(
            grid_size=(self.config['n'], self.config['n']),
            args=self.config,
            priority_grid=priority_grid_arr
        )

        # Get algorithm class and instantiate
        algo_class = get_algorithm_class(self.config['algorithm'])

        # Set trial ID for tree visualization if enabled
        if self.config.get('tree_visualization', False):
            trial_id = f"trial_{self.config.get('random_seed', 'unknown')}_n{self.config.get('n', 'unknown')}"
        else:
            trial_id = None

        # Initialize algorithm
        self.algorithm = algo_class(self.env, args=self.config)

        if trial_id and hasattr(self.algorithm, 'trial_id'):
            self.algorithm.trial_id = trial_id

    def run(self) -> int:
        """
        Run the experiment.

        Returns:
            Number of points achieved in the final configuration
        """
        if self.env is None or self.algorithm is None:
            self.setup()

        # Import utilities here
        from src.algos.mcts.utils import select_outermost_with_tiebreaker

        start = time.time()
        n = self.config['n']

        state = self.env.get_initial_state()
        num_of_points = 0

        while True:
            if self.config.get('display_state', False):
                print("---------------------------")
                print(f"Number of points: {num_of_points}")
                print(state)

            valid_moves = self.env.get_valid_moves(state)
            value, is_terminal = self.env.get_value_and_terminated(state, valid_moves)

            if is_terminal:
                print("*******************************************************************")
                print(f"Trial Terminated with {num_of_points} points. Final valid configuration:")
                print(state)
                self.env.display_state(state, mcts_probs)
                end = time.time()
                print(f"Time: {end - start:.6f} sec")

                # Record results to table
                self.env.record_to_table(
                    terminal_num_points=num_of_points,
                    start_time=start,
                    end_time=end,
                    time_used=end - start
                )

                # Save tree visualization snapshots if enabled
                if self.config.get('tree_visualization', False) and hasattr(self.algorithm, 'snapshots') and self.algorithm.snapshots:
                    print(f"Collected {len(self.algorithm.snapshots)} tree snapshots for this trial")
                    # Check if algorithm has global_trial_data attribute (MCTS-specific)
                    if hasattr(self.algorithm.__class__, 'global_trial_data'):
                        print(f"Total global snapshots so far: {len(self.algorithm.__class__.global_trial_data)}")

                break

            # Get algorithm action probabilities
            mcts_probs = self.algorithm.search(state)

            # Use outermost-priority selector to pick action
            action = select_outermost_with_tiebreaker(mcts_probs, n)

            # Display state and probabilities
            if self.config.get('display_state', False):
                self.env.display_state(state, mcts_probs)

            # Apply action
            num_of_points += 1
            state = self.env.get_next_state(state, action)

        # Always return the number of points (logging_mode controls whether to print, not return)
        return num_of_points


def run_experiment(config: Dict[str, Any]) -> int:
    """
    Run a single experiment with the given configuration.

    This is a convenience function that creates and runs an ExperimentRunner.

    Args:
        config: Configuration dictionary

    Returns:
        Number of points achieved

    Example:
        config = {
            'algorithm': 'MCTS',
            'environment': 'N3il',
            'n': 10,
            'num_searches': 1000,
            ...
        }
        num_points = run_experiment(config)
    """
    runner = ExperimentRunner(config)
    return runner.run()


# Backward compatibility alias
evaluate = run_experiment
