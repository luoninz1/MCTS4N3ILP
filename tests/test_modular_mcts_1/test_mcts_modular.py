"""Test script for my experiment."""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.experiment import ExperimentRunner
from tests.test_modular_mcts_1.config import get_mcts_config

def main():
    # Configure experiment
    config = get_mcts_config(n=20, random_seed=42)

    # Set output directories
    config['table_dir'] = os.path.join(os.path.dirname(__file__), 'results')
    config['figure_dir'] = os.path.join(os.path.dirname(__file__), 'figures')

    # Run experiment
    runner = ExperimentRunner(config)
    num_points = runner.run()

    print(f"Experiment completed: {num_points} points")

if __name__ == "__main__":
    main()