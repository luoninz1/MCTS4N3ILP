# MCTS4N3ILP

Monte Carlo Tree Search for No-Three-in-Line Problem

## Overview

This project implements various Monte Carlo Tree Search (MCTS) algorithms for solving the No-Three-in-Line Problem (N3ILP). The codebase is designed with modularity in mind, making it easy to experiment with different algorithms and environments.

## Project Structure

```
MCTS4N3ILP/
├── src/
│   ├── algos/              # Algorithm implementations
│   │   └── mcts/           # MCTS variants
│   │       ├── tree_search.py      # MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS
│   │       ├── node.py             # Node classes (Node, Node_Compressed, etc.)
│   │       ├── simulation.py       # Simulation strategies
│   │       ├── utils.py            # Utility functions
│   │       ├── visualization.py    # Tree visualization
│   │       └── evaluation.py       # [DEPRECATED] Use src/experiment instead
│   │
│   ├── envs/               # Environment implementations
│   │   └── n3il/           # No-Three-in-Line environment
│   │       ├── n3il_env.py             # Base environment
│   │       ├── n3il_symmetry_env.py    # Environment with D4 symmetry
│   │       ├── rewards.py              # Reward functions
│   │       ├── priority.py             # Priority calculations
│   │       ├── symmetry.py             # Symmetry operations
│   │       ├── visualization.py        # Grid visualization
│   │       └── logging.py              # Result logging
│   │
│   └── experiment/         # Experiment orchestration [NEW]
│       ├── __init__.py
│       ├── runner.py       # ExperimentRunner class
│       └── registry.py     # Algorithm/environment registry
│
├── tests/                  # Test suites
│   ├── test_modular_mcts/  # Example test suite
│   │   ├── config.py       # Configuration presets
│   │   ├── test_mcts_modular.py
│   │   ├── results/        # Experiment results (auto-generated)
│   │   └── figures/        # Visualization outputs (auto-generated)
│   │
│   ├── test_mcts_smallest_complete_v2/
│   └── test_mcts_largest_complete_v2/
│
├── environment.yml         # Conda environment specification
├── REFACTORING_SUMMARY.md  # Detailed refactoring notes
└── README.md               # This file
```

## Available Algorithms

- **MCTS**: Standard Monte Carlo Tree Search
- **ParallelMCTS**: Multi-threaded MCTS with virtual loss
- **LeafChildParallelMCTS**: Leaf-parallel and child-parallel MCTS variants
- **MCGS**: Monte Carlo Graph Search

## Available Environments

- **N3il**: Base No-Three-in-Line environment
- **N3il_with_symmetry**: Environment with D4 symmetry reduction

## Installation

### Using Conda

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate mcts4n3ilp
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n mcts4n3ilp python=3.9
conda activate mcts4n3ilp

# Install required packages
pip install numpy matplotlib tqdm
```

## Quick Start

### Running an Experiment

```python
from src.experiment import ExperimentRunner

# Define configuration
config = {
    'algorithm': 'MCTS',           # Algorithm to use
    'environment': 'N3il',         # Environment to use
    'n': 10,                       # Grid size (10x10)
    'num_searches': 1000,          # Number of MCTS searches
    'C': 1.41,                     # Exploration constant
    'random_seed': 42,             # Random seed
    'display_state': True,         # Display board state
    'logging_mode': True,          # Enable logging
}

# Run experiment
runner = ExperimentRunner(config)
num_points = runner.run()
print(f"Achieved {num_points} points")
```

### Using Configuration Presets

```python
from src.experiment import ExperimentRunner
from tests.test_modular_mcts.config import get_mcts_config, set_output_directories

# Get preset configuration
config = get_mcts_config(n=10, random_seed=42, num_searches_multiplier=100)
config = set_output_directories(config, test_name="my_experiment")

# Run experiment
runner = ExperimentRunner(config)
num_points = runner.run()
```

## Creating New Tests

To conduct new experiments, follow these steps:

### 1. Create a New Test Directory

```bash
mkdir tests/test_my_experiment
cd tests/test_my_experiment
```

### 2. Create `config.py`

Define your experiment configurations:

```python
"""Configuration for my experiment."""

def get_base_config():
    """Get base configuration."""
    return {
        'process_bar': True,
        'display_state': True,
        'logging_mode': True,
    }

def get_my_experiment_config(n, random_seed=0):
    """Get experiment-specific configuration."""
    config = get_base_config()
    config.update({
        'algorithm': 'MCTS',        # Choose algorithm
        'environment': 'N3il',      # Choose environment
        'n': n,
        'C': 1.41,
        'num_searches': 100 * (n ** 2),
        'random_seed': random_seed,
        # Add other parameters...
    })
    return config
```

### 3. Create Test Script

Create `test_my_experiment.py`:

```python
"""Test script for my experiment."""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.experiment import ExperimentRunner
from tests.test_my_experiment.config import get_my_experiment_config

def main():
    # Configure experiment
    config = get_my_experiment_config(n=10, random_seed=42)

    # Set output directories
    config['table_dir'] = os.path.join(os.path.dirname(__file__), 'results')
    config['figure_dir'] = os.path.join(os.path.dirname(__file__), 'figures')

    # Run experiment
    runner = ExperimentRunner(config)
    num_points = runner.run()

    print(f"Experiment completed: {num_points} points")

if __name__ == "__main__":
    main()
```

### 4. Run Your Test

```bash
cd tests/test_my_experiment
python test_my_experiment.py
```

Results will be saved in:
- `tests/test_my_experiment/results/` - CSV tables with experiment data
- `tests/test_my_experiment/figures/` - Visualization outputs

## Running Existing Tests

### Modular MCTS Test Suite

```bash
cd tests/test_modular_mcts

# Basic test (grid sizes 3-10, 3 trials each)
python test_mcts_modular.py --start 3 --end 11 --repeat 3

# Quick test (fewer searches)
python test_mcts_modular.py --config quick_test --start 3 --end 6

# Test with symmetry
python test_mcts_modular.py --config symmetry --repeat 10

# Custom search budget
python test_mcts_modular.py --searches 200 --repeat 5

# Available configs: basic, symmetry, compressed, parallel, quick_test, medium_grid, large_grid
```

### Other Test Suites

```bash
# Smallest complete test
cd tests/test_mcts_smallest_complete_v2
python smallest_complete_v2.py --start 7 --end 100 --step 100 --repeat 10

# Largest complete test
cd tests/test_mcts_largest_complete_v2
python largest_complete_v2.py --start 3 --end 50 --step 1 --repeat 10
```

## Configuration Options

### Algorithm Parameters

```python
config = {
    # Algorithm selection
    'algorithm': 'MCTS',  # 'MCTS', 'ParallelMCTS', 'LeafChildParallelMCTS', 'MCGS'

    # MCTS parameters
    'num_searches': 1000,           # Number of search iterations
    'C': 1.41,                      # UCT exploration constant
    'node_compression': False,      # Use compressed node representation

    # Parallel parameters (for ParallelMCTS)
    'num_workers': 4,               # Number of parallel workers
    'virtual_loss': 1.0,            # Virtual loss magnitude

    # Leaf/Child parallel (for LeafChildParallelMCTS)
    'simulations_per_leaf': 1,      # Simulations per leaf node
    'child_parallel': True,         # Enable child-parallel expansion

    # MCGS parameters
    'gamma': 0.99,                  # Discount factor
}
```

### Environment Parameters

```python
config = {
    # Environment selection
    'environment': 'N3il',  # 'N3il' or 'N3il_with_symmetry'

    # Grid parameters
    'n': 10,                        # Grid size (n x n)

    # Symmetry parameters (for N3il_with_symmetry)
    'max_level_to_use_symmetry': 1, # Max depth for symmetry (0 = disabled)

    # Priority parameters
    'TopN': 10,                     # Number of top-priority moves to consider
    'simulate_with_priority': False, # Use priority-based simulation
}
```

### Experiment Parameters

```python
config = {
    # Reproducibility
    'random_seed': 42,

    # Logging and visualization
    'display_state': True,          # Print board states
    'logging_mode': True,           # Enable result logging
    'process_bar': True,            # Show progress bar
    'tree_visualization': False,    # Enable tree visualization
    'pause_at_each_step': False,    # Pause for user input

    # Output directories
    'table_dir': './results',       # CSV output directory
    'figure_dir': './figures',      # Figure output directory
}
```

## Available Configuration Presets

From `tests/test_modular_mcts/config.py`:

- `get_mcts_config()` - Basic MCTS
- `get_mcts_with_symmetry_config()` - MCTS with symmetry
- `get_mcts_compressed_config()` - MCTS with node compression
- `get_parallel_mcts_config()` - Parallel MCTS
- `get_mcts_with_priority_config()` - MCTS with priority simulation
- `get_leaf_child_parallel_config()` - Leaf/child parallel MCTS
- `get_mcgs_config()` - Monte Carlo Graph Search

Preset shortcuts:
- `quick_test` - Fast testing (10x searches)
- `medium_grid` - Standard experiments (100x searches)
- `large_grid` - Thorough experiments (200x searches)

## Advanced Usage

### Adding a New Algorithm

1. Implement your algorithm in `src/algos/your_algo/`
2. Register it in `src/experiment/registry.py`:

```python
from src.algos.your_algo import YourAlgorithm
ALGORITHM_REGISTRY['YourAlgo'] = YourAlgorithm
```

3. Use it in experiments:

```python
config = {
    'algorithm': 'YourAlgo',
    'environment': 'N3il',
    # ... other parameters
}
```

### Adding a New Environment

1. Implement your environment in `src/envs/your_env/`
2. Register it in `src/experiment/registry.py`:

```python
from src.envs.your_env import YourEnvironment
ENVIRONMENT_REGISTRY['YourEnv'] = YourEnvironment
```

3. Use it in experiments:

```python
config = {
    'algorithm': 'MCTS',
    'environment': 'YourEnv',
    # ... other parameters
}
```

### List Available Options

```python
from src.experiment import list_algorithms, list_environments

print("Available algorithms:", list_algorithms())
print("Available environments:", list_environments())
```

## Backward Compatibility

Old code using the deprecated interface will continue to work:

```python
# Old way (still works)
from src.algos.mcts import evaluate

config = {...}
num_points = evaluate(config)

# New way (recommended)
from src.experiment import ExperimentRunner

config = {...}
runner = ExperimentRunner(config)
num_points = runner.run()
```

## Key Features

- **Modular Design**: Easy to swap algorithms and environments
- **Configuration-Driven**: Centralized parameter management
- **Extensible**: Simple registry system for adding new components
- **Reproducible**: Seed-based randomization for consistent results
- **Parallel Processing**: Multi-threaded MCTS variants
- **Visualization**: Tree and grid visualization support
- **Experiment Tracking**: Automatic result logging and CSV export

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mcts4n3ilp,
  title = {MCTS4N3ILP: Monte Carlo Tree Search for No-Three-in-Line Problem},
  author = {Luoning Zhang},
  year = {2025},
  url = {https://github.com/luoninz1/mcts4n3ilp}
}
```

## License

See [LICENSE](LICENSE) file for details.

## References

For more details on the refactoring and modular design, see [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md).
