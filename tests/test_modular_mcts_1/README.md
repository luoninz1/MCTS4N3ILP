## Modular MCTS Test Suite

This directory contains a clean, modular test suite for the No-Three-In-Line MCTS implementation.

### Directory Structure

```
test_modular_mcts/
├── README.md                 # This file
├── config.py                 # Configuration presets and parameter management
├── test_mcts_modular.py      # Main test script
├── results/                  # Experiment results (CSV files)
└── figures/                  # Generated figures
```

### Configuration System

The test suite uses a modular configuration system defined in `config.py`. This allows for:

1. **Preset Configurations**: Pre-defined parameter sets for common experiments
2. **Custom Configurations**: Build configurations programmatically
3. **Parameter Isolation**: All parameters in one place for easy management

### Available Configurations

#### Basic Configurations
- `get_mcts_config()` - Standard MCTS
- `get_mcts_with_symmetry_config()` - MCTS with D4 symmetry
- `get_mcts_compressed_config()` - MCTS with node compression
- `get_parallel_mcts_config()` - Parallel MCTS
- `get_mcts_with_priority_config()` - MCTS with priority-based simulation
- `get_leaf_child_parallel_config()` - Leaf/Child parallel MCTS
- `get_mcgs_config()` - Monte Carlo Graph Search

#### Presets
- `small_grid` - 50x searches per grid
- `medium_grid` - 100x searches per grid
- `large_grid` - 200x searches per grid
- `quick_test` - 10x searches per grid (for rapid testing)
- `symmetry_test` - MCTS with symmetry enabled
- `parallel_test` - Parallel MCTS with 4 workers

### Usage Examples

#### Basic Usage
```bash
# Test grid sizes 3-10 with basic MCTS
python test_mcts_modular.py --start 3 --end 11 --repeat 3

# Quick test with fewer searches
python test_mcts_modular.py --start 3 --end 6 --repeat 1 --config quick_test

# Test with symmetry
python test_mcts_modular.py --start 3 --end 11 --config symmetry

# Custom search multiplier
python test_mcts_modular.py --start 3 --end 11 --searches 200
```

#### Advanced Usage
```bash
# Test larger grids with more searches
python test_mcts_modular.py --start 10 --end 21 --step 5 --config large_grid --repeat 5

# Parallel MCTS testing
python test_mcts_modular.py --start 3 --end 11 --config parallel --repeat 3

# Node compression testing
python test_mcts_modular.py --start 3 --end 11 --config compressed
```

### Command-Line Arguments

- `--start N`: Starting grid size (default: 3)
- `--end N`: Ending grid size, exclusive (default: 11)
- `--step N`: Step size for grid sizes (default: 1)
- `--repeat N`: Number of trials per grid size (default: 3)
- `--config NAME`: Configuration preset to use (default: basic)
- `--searches N`: Search multiplier, total searches = N * n² (default: 100)

### Ablation Study Examples

#### Test different algorithms
```bash
# Compare basic vs symmetry
python test_mcts_modular.py --config basic --repeat 10
python test_mcts_modular.py --config symmetry --repeat 10

# Compare different search budgets
python test_mcts_modular.py --searches 50 --repeat 10
python test_mcts_modular.py --searches 100 --repeat 10
python test_mcts_modular.py --searches 200 --repeat 10
```

### Output

Results are saved to:
- **CSV files**: `results/{test_name}/experiment_results.csv`
- **Figures**: `figures/{test_name}/`

Each CSV file contains:
- All configuration parameters
- Number of points achieved
- Runtime information
- Random seed for reproducibility

### Customization

To add a new configuration:

1. Add a configuration function to `config.py`:
```python
def get_my_custom_config(n, random_seed=0):
    config = get_base_config()
    config.update({
        'n': n,
        'random_seed': random_seed,
        # ... your parameters
    })
    return config
```

2. Use it in your tests:
```python
config = get_my_custom_config(n=10, random_seed=42)
result = run_single_experiment(config)
```
