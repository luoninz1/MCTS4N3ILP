"""
Experiment module for running algorithm evaluations.

This module provides a flexible framework for running experiments with
different algorithms and environments.

Usage:
    from src.experiment import ExperimentRunner, run_experiment

    config = {
        'algorithm': 'MCTS',
        'environment': 'N3il',
        'n': 10,
        'num_searches': 1000,
        ...
    }

    # Option 1: Use the runner class
    runner = ExperimentRunner(config)
    result = runner.run()

    # Option 2: Use the convenience function
    result = run_experiment(config)

    # Option 3: Backward compatibility
    from src.experiment import evaluate
    result = evaluate(config)
"""

from src.experiment.runner import ExperimentRunner, run_experiment, evaluate
from src.experiment.registry import (
    get_algorithm_class,
    get_environment_class,
    list_algorithms,
    list_environments,
    ALGORITHM_REGISTRY,
    ENVIRONMENT_REGISTRY
)

__all__ = [
    # Main experiment classes and functions
    'ExperimentRunner',
    'run_experiment',
    'evaluate',  # Backward compatibility

    # Registry functions
    'get_algorithm_class',
    'get_environment_class',
    'list_algorithms',
    'list_environments',
    'ALGORITHM_REGISTRY',
    'ENVIRONMENT_REGISTRY',
]
