# Modular Design Improvements

This document summarizes the modular design improvements made to the MCTS4N3ILP project to enhance compatibility with different algorithms and environments.

## Summary of Changes

### 1. Created `src/experiment/` Module (Experiment Orchestration Layer)

**Problem**: The `evaluation.py` file was located in `src/algos/mcts/`, making it algorithm-specific and tightly coupled.

**Solution**: Created a new `src/experiment/` module that serves as an orchestration layer between algorithms and environments.

**Files Created**:
- `src/experiment/__init__.py` - Package interface
- `src/experiment/runner.py` - `ExperimentRunner` class and `run_experiment()` function
- `src/experiment/registry.py` - Algorithm and environment registry system

**Benefits**:
- ✅ Clean separation: `algos ← experiment → envs`
- ✅ Easy to add new algorithms (just register them)
- ✅ Easy to add new environments (just register them)
- ✅ Configuration-driven experiment design
- ✅ Backward compatible with existing code

### 2. Removed Environment-Specific Import from `simulation.py`

**Problem**: `src/algos/mcts/simulation.py` imported `get_value_nb` directly from `src.envs.n3il.rewards`, making it incompatible with future environments that might have different reward functions.

**Solution**: Made simulation functions environment-agnostic by:
1. Adding a default value function in `simulation.py`
2. Adding optional `value_fn` parameter to `simulate_nb()` and `simulate_with_priority_nb()`
3. Storing environment's value function in the environment object (`self.value_fn`)
4. Node classes now pass the environment's value function to simulation functions

**Changes Made**:

#### `src/algos/mcts/simulation.py`
```python
# Before
from src.envs.n3il.rewards import get_value_nb

def simulate_nb(state, row_count, column_count, pts_upper_bound):
    # ...
    return get_value_nb(state, pts_upper_bound)

# After
def _default_value_fn(state, pts_upper_bound):
    """Default value function for point-placement games."""
    return np.sum(state) / pts_upper_bound

def simulate_nb(state, row_count, column_count, pts_upper_bound, value_fn=None):
    # ...
    if value_fn is None:
        return _default_value_fn(state, pts_upper_bound)
    else:
        return value_fn(state, pts_upper_bound)
```

#### `src/envs/n3il/n3il_env.py`
```python
# Added to __init__
self.value_fn = get_value_nb  # Can be overridden by subclasses
```

#### `src/algos/mcts/node.py`
```python
# Updated simulate() in Node, Node_Compressed, and LeafChildParallelNode
def simulate(self):
    tmp = self.state.copy()
    value_fn = getattr(self.game, 'value_fn', None)  # Get from environment

    return simulate_nb(tmp,
                      self.game.row_count,
                      self.game.column_count,
                      self.game.pts_upper_bound,
                      value_fn)  # Pass to simulation
```

**Benefits**:
- ✅ No hard dependency on specific environment reward functions
- ✅ Environments can provide custom value functions
- ✅ Backward compatible (uses default if not provided)
- ✅ Follows dependency inversion principle

## Architecture

### Before
```
src/
├── algos/mcts/
│   ├── evaluation.py    # Mixed concerns (orchestration + MCTS)
│   └── simulation.py    # Imports from src.envs.n3il (tight coupling)
└── envs/n3il/
```

### After
```
src/
├── algos/mcts/
│   ├── simulation.py    # Environment-agnostic (no direct imports from envs)
│   └── node.py          # Uses env.value_fn if available
│
├── envs/n3il/
│   └── n3il_env.py      # Provides self.value_fn attribute
│
└── experiment/          # NEW: Orchestration layer
    ├── __init__.py
    ├── runner.py        # ExperimentRunner class
    └── registry.py      # Dynamic algo/env loading
```

## How to Add New Environments

### Step 1: Implement Your Environment

```python
# src/envs/my_env/my_env.py
import numpy as np
from numba import njit

@njit
def my_custom_value_fn(state, pts_upper_bound):
    """Custom value function for my environment."""
    # Your custom logic here
    return custom_score / pts_upper_bound

class MyEnvironment:
    def __init__(self, grid_size, args, priority_grid=None):
        self.row_count, self.column_count = grid_size
        self.action_size = self.row_count * self.column_count
        self.pts_upper_bound = 100  # Your upper bound
        self.args = args

        # Set custom value function
        self.value_fn = my_custom_value_fn  # <-- Key: provide value_fn

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), np.uint8)

    def get_valid_moves(self, state):
        # Your logic
        pass

    def get_value_and_terminated(self, state, valid_moves):
        # Your logic
        pass

    # ... other required methods
```

### Step 2: Register Your Environment

```python
# src/experiment/registry.py
from src.envs.my_env.my_env import MyEnvironment

ENVIRONMENT_REGISTRY['MyEnv'] = MyEnvironment
```

### Step 3: Use It in Experiments

```python
from src.experiment import ExperimentRunner

config = {
    'algorithm': 'MCTS',
    'environment': 'MyEnv',  # <-- Your new environment
    'n': 10,
    # ... other parameters
}

runner = ExperimentRunner(config)
result = runner.run()
```

## How to Add New Algorithms

### Step 1: Implement Your Algorithm

```python
# src/algos/my_algo/my_algo.py
class MyAlgorithm:
    def __init__(self, game, args):
        self.game = game  # Environment instance
        self.args = args

    def search(self, state):
        """
        Perform search and return action probabilities.

        Returns:
            np.ndarray: Probability distribution over actions
        """
        # Your algorithm logic
        action_probs = np.zeros(self.game.action_size)
        # ... compute probabilities
        return action_probs
```

### Step 2: Register Your Algorithm

```python
# src/experiment/registry.py
from src.algos.my_algo.my_algo import MyAlgorithm

ALGORITHM_REGISTRY['MyAlgo'] = MyAlgorithm
```

### Step 3: Use It in Experiments

```python
from src.experiment import ExperimentRunner

config = {
    'algorithm': 'MyAlgo',  # <-- Your new algorithm
    'environment': 'N3il',
    'n': 10,
    # ... other parameters
}

runner = ExperimentRunner(config)
result = runner.run()
```

## Backward Compatibility

All existing code continues to work:

```python
# Old way (still works)
from src.algos.mcts import evaluate

config = {...}
result = evaluate(config)

# New way (recommended)
from src.experiment import ExperimentRunner

config = {...}
runner = ExperimentRunner(config)
result = runner.run()
```

## Testing New Environments/Algorithms

Create a test directory following the pattern:

```
tests/
└── test_my_experiment/
    ├── __init__.py
    ├── config.py           # Configuration functions
    └── test_my_exp.py      # Test script
```

Example `config.py`:
```python
def get_my_config(n, random_seed=0):
    return {
        'algorithm': 'MyAlgo',
        'environment': 'MyEnv',
        'n': n,
        'random_seed': random_seed,
        # ... other parameters
    }
```

Example `test_my_exp.py`:
```python
from src.experiment import ExperimentRunner
from tests.test_my_experiment.config import get_my_config

config = get_my_config(n=10, random_seed=42)
runner = ExperimentRunner(config)
result = runner.run()
print(f"Result: {result}")
```

## Summary of Benefits

1. **Modularity**: Clear separation between algorithms, environments, and orchestration
2. **Extensibility**: Easy to add new algorithms and environments
3. **Flexibility**: Environments can provide custom value functions
4. **Maintainability**: No circular dependencies or tight coupling
5. **Testability**: Each component can be tested independently
6. **Backward Compatible**: Existing code continues to work

## Migration Guide

If you have custom code that imports from the old locations:

### Old Import (deprecated but still works)
```python
from src.algos.mcts import evaluate
```

### New Import (recommended)
```python
from src.experiment import ExperimentRunner, run_experiment
```

### Old Simulation Import (deprecated)
```python
from src.envs.n3il.rewards import get_value_nb
```

### New Approach (recommended)
```python
# In your environment class
from src.envs.n3il.rewards import get_value_nb

class MyEnv:
    def __init__(self, ...):
        self.value_fn = get_value_nb  # Set as attribute
```

## Future Improvements

Potential future enhancements:
- Add type hints for better IDE support
- Create abstract base classes for algorithms and environments
- Add validation for configuration dictionaries
- Support for config file loading (YAML/JSON)
- Plugin system for dynamic loading of algorithms/environments
