"""
Modular MCTS Test Suite

This test demonstrates the modular MCTS implementation with clean
separation of concerns and configurable parameters.
"""

import sys
import os
import argparse
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# New import from experiment module (backward compatible import also works)
from src.experiment import ExperimentRunner, run_experiment
from src.algos.mcts import MCTS  # For clearing global data
from tests.test_modular_mcts.config import (
    get_mcts_config,
    get_mcts_with_symmetry_config,
    get_mcts_compressed_config,
    get_parallel_mcts_config,
    get_preset_config,
    set_output_directories
)


def run_single_experiment(config):
    """
    Run a single experiment with the given configuration.

    Args:
        config (dict): Experiment configuration

    Returns:
        int: Number of points achieved
    """
    # New way: use ExperimentRunner or run_experiment
    # Both approaches work:
    # return run_experiment(config)  # Functional approach
    runner = ExperimentRunner(config)
    return runner.run()  # Object-oriented approach


def run_experiment_suite(n_list, config_func, test_name, repeat=1):
    """
    Run a suite of experiments across multiple grid sizes.

    Args:
        n_list (iterable): List of grid sizes to test
        config_func (callable): Function that takes (n, seed) and returns config dict
        test_name (str): Name of the test suite
        repeat (int): Number of trials per grid size

    Returns:
        dict: Results dictionary mapping n -> list of point counts
    """
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")

    all_results = {}

    for n in n_list:
        print(f"\n=== Grid size n={n} ===")
        results_for_n = []

        for trial in range(repeat):
            print(f"Trial {trial+1}/{repeat} for n={n}...", end=" ")

            # Get configuration for this trial
            config = config_func(n, trial)
            config = set_output_directories(config, test_name)

            # Run experiment
            num_points = run_single_experiment(config)
            results_for_n.append(num_points)
            print(f"Points: {num_points}")

        all_results[n] = results_for_n

        # Print summary for this grid size
        print(f"\n--- Summary for n={n} ---")
        print(f"Results: {results_for_n}")
        print(f"Min: {min(results_for_n)}, Max: {max(results_for_n)}, "
              f"Avg: {sum(results_for_n)/len(results_for_n):.2f}")

    return all_results


def print_final_summary(all_results):
    """Print final summary of all experiments."""
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    for n, results in all_results.items():
        freq = Counter(results)
        print(f"\nGrid size n={n}:")
        print(f"  Results: {results}")
        print(f"  Min: {min(results)}, Max: {max(results)}, "
              f"Avg: {sum(results)/len(results):.4f}")
        print(f"  Frequency: {dict(freq)}")

        # Check for known optimal solutions
        if n == 3 and 4 in freq:
            optimal_count = freq[4]
            print(f"  ✓ Found optimal 4-point solution: "
                  f"{optimal_count}/{len(results)} times "
                  f"({100*optimal_count/len(results):.1f}%)")


def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(
        description="Run modular MCTS tests for No-Three-In-Line problem"
    )
    parser.add_argument(
        "--start", type=int, default=10,
        help="Starting grid size (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, default=11,
        help="Ending grid size (exclusive)"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Step size for grid sizes"
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="Number of trials per grid size"
    )
    parser.add_argument(
        "--config", type=str, default="basic",
        choices=[
            "basic", "symmetry", "compressed", "parallel",
            "quick_test", "medium_grid", "large_grid"
        ],
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--searches", type=int, default=100,
        help="Search multiplier (searches = multiplier * n^2)"
    )

    args = parser.parse_args()

    # Generate grid size list
    n_list = range(args.start, args.end, args.step)

    # Clear any previous global data
    MCTS.clear_global_data()

    # Select configuration function based on preset
    if args.config == "basic":
        config_func = lambda n, seed: get_mcts_config(
            n, seed, num_searches_multiplier=args.searches
        )
        test_name = f"basic_mcts_searches{args.searches}"

    elif args.config == "symmetry":
        config_func = lambda n, seed: get_mcts_with_symmetry_config(
            n, seed, num_searches_multiplier=args.searches, max_symmetry_level=1
        )
        test_name = f"symmetry_mcts_searches{args.searches}"

    elif args.config == "compressed":
        config_func = lambda n, seed: get_mcts_compressed_config(
            n, seed, num_searches_multiplier=args.searches
        )
        test_name = f"compressed_mcts_searches{args.searches}"

    elif args.config == "parallel":
        config_func = lambda n, seed: get_parallel_mcts_config(
            n, seed, num_searches_multiplier=args.searches, num_workers=4
        )
        test_name = f"parallel_mcts_searches{args.searches}"

    else:  # Preset configurations
        config_func = lambda n, seed: get_preset_config(args.config, n, seed)
        test_name = f"preset_{args.config}"

    # Run experiments
    all_results = run_experiment_suite(
        n_list=n_list,
        config_func=config_func,
        test_name=test_name,
        repeat=args.repeat
    )

    # Print summary
    print_final_summary(all_results)

    print(f"\n{'='*70}")
    print(f"  Test completed: {test_name}")
    print(f"  Results saved to: tests/test_modular_mcts/results/{test_name}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
