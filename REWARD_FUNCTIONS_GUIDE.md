# Reward Functions Guide

This guide explains how to configure and customize reward/value functions in the MCTS4N3ILP project.

## Overview

The reward function (also called value function) determines how terminal states are scored during MCTS simulations. **All reward functions are now centralized in a single location**: `src/rewards.py`.

## Quick Start

### 1. Using Built-in Reward Functions

Simply specify the reward function name in your config:

```python
from src.experiment import ExperimentRunner

config = {
    'algorithm': 'MCTS',
    'environment': 'N3il',
    'n': 10,
    'value_function': 'point_count',  # <-- Choose reward function here
    # ... other parameters
}

runner = ExperimentRunner(config)
result = runner.run()
```

### 2. Available Built-in Functions

| Function Name | Description | Use Case |
|--------------|-------------|----------|
| `point_count` | Linear: `points / max_points` | **Default**, balanced exploration |
| `point_count_squared` | Quadratic: `(points / max_points)²` | Aggressive, prioritizes high scores |
| `point_count_log` | Logarithmic: `log(1 + points) / log(1 + max)` | Conservative, diminishing returns |

### 3. List Available Functions

```python
from src.rewards import list_value_functions

print("Available reward functions:", list_value_functions())
# Output: ['point_count', 'point_count_squared', 'point_count_log']
```

## Creating Custom Reward Functions

### Method 1: Add to Central Registry (Recommended)

Edit `src/rewards.py` and add your custom function:

```python
# src/rewards.py

@njit(cache=True, nogil=True)
@register_value_function('my_custom_reward')
def my_custom_reward(state, pts_upper_bound):
    """
    Custom reward function.

    Args:
        state: 2D numpy array of the board
        pts_upper_bound: Maximum expected points

    Returns:
        float: Normalized reward between 0 and 1
    """
    num_points = np.sum(state)

    # Example: Exponential reward
    return (np.exp(num_points / pts_upper_bound) - 1) / (np.e - 1)
```

Then use it:

```python
config = {
    'value_function': 'my_custom_reward',  # Use your custom function
    # ... other parameters
}
```

### Method 2: Pass Function Directly

You can also pass a function directly (must be a numba-compiled function):

```python
from numba import njit
import numpy as np

@njit(cache=True, nogil=True)
def custom_reward(state, pts_upper_bound):
    """My custom reward logic."""
    num_points = np.sum(state)
    return num_points / pts_upper_bound

config = {
    'value_function': custom_reward,  # Pass function directly
    # ... other parameters
}
```

## Configuration Examples

### Example 1: Basic Configuration with Different Rewards

```python
from tests.test_modular_mcts.config import get_mcts_config

# Standard linear reward
config_linear = get_mcts_config(n=10, random_seed=42)
config_linear['value_function'] = 'point_count'

# Aggressive quadratic reward
config_quad = get_mcts_config(n=10, random_seed=42)
config_quad['value_function'] = 'point_count_squared'

# Conservative logarithmic reward
config_log = get_mcts_config(n=10, random_seed=42)
config_log['value_function'] = 'point_count_log'
```

### Example 2: Using the Helper Function

```python
from tests.test_modular_mcts.config import get_mcts_with_custom_reward_config

# Automatically sets value_function in config
config = get_mcts_with_custom_reward_config(
    n=10,
    random_seed=42,
    value_function='point_count_squared'
)
```

### Example 3: Comparing Reward Functions

```python
from src.experiment import ExperimentRunner

reward_functions = ['point_count', 'point_count_squared', 'point_count_log']
results = {}

for reward_fn in reward_functions:
    config = {
        'algorithm': 'MCTS',
        'environment': 'N3il',
        'n': 10,
        'value_function': reward_fn,
        'num_searches': 1000,
        'random_seed': 42,
        # ... other params
    }

    runner = ExperimentRunner(config)
    results[reward_fn] = runner.run()

print("Results by reward function:")
for fn, points in results.items():
    print(f"  {fn}: {points} points")
```

## Implementation Details

### Architecture

```
src/
├── rewards.py              # Centralized reward functions (SINGLE SOURCE OF TRUTH)
│   ├── point_count_value()
│   ├── point_count_squared_value()
│   ├── point_count_log_value()
│   └── get_value_function()  # Registry lookup
│
├── envs/n3il/
│   ├── n3il_env.py        # Reads value_fn from config
│   └── rewards.py         # [DEPRECATED] Re-exports from src/rewards.py
│
└── algos/mcts/
    └── simulation.py       # Uses env.value_fn (passed from environment)
```

### Data Flow

```
1. Config specifies value_function
   ↓
2. Environment reads from config and looks up function
   ↓
3. Environment stores as self.value_fn
   ↓
4. Node.simulate() gets value_fn from environment
   ↓
5. Simulation applies value_fn to final state
```

### Why This Design?

**Before**: Reward function was duplicated in two places:
- ❌ `src/envs/n3il/rewards.py`
- ❌ `src/algos/mcts/simulation.py` (had default copy)

**After**: Single source of truth:
- ✅ `src/rewards.py` - All reward functions here
- ✅ Configurable via config file
- ✅ No duplication
- ✅ Easy to experiment

## Advanced Usage

### Creating Domain-Specific Reward Functions

For different problem domains, you might want different reward structures:

```python
# For sparse reward problems
@njit(cache=True, nogil=True)
@register_value_function('sparse_reward')
def sparse_reward(state, pts_upper_bound):
    """Only reward if threshold is met."""
    num_points = np.sum(state)
    threshold = pts_upper_bound * 0.8

    if num_points >= threshold:
        return 1.0
    else:
        return 0.0

# For shaped rewards
@njit(cache=True, nogil=True)
@register_value_function('shaped_reward')
def shaped_reward(state, pts_upper_bound):
    """Combination of linear + bonus for high scores."""
    num_points = np.sum(state)
    base_reward = num_points / pts_upper_bound

    # Bonus for exceeding 90% of upper bound
    if num_points > 0.9 * pts_upper_bound:
        bonus = 0.2
    else:
        bonus = 0.0

    return min(1.0, base_reward + bonus)
```

### Dynamic Reward Functions

You can also create reward functions that depend on grid size or other parameters:

```python
def create_size_dependent_reward(power=1.0):
    """Factory function for size-dependent rewards."""

    @njit(cache=True, nogil=True)
    def reward_fn(state, pts_upper_bound):
        num_points = np.sum(state)
        normalized = num_points / pts_upper_bound
        return normalized ** power

    return reward_fn

# Use in config
config = {
    'value_function': create_size_dependent_reward(power=1.5),
    # ... other params
}
```

## Best Practices

1. **Always use `@njit` decorator** for performance
2. **Keep functions pure** - no side effects
3. **Return values between 0 and 1** for consistency
4. **Document your reward function** - explain the intuition
5. **Test with different seeds** - ensure reward drives desired behavior

## Troubleshooting

### Issue: "Unknown value function: xyz"

**Solution**: Make sure the function is registered:
```python
from src.rewards import list_value_functions
print(list_value_functions())  # Check if your function is listed
```

### Issue: Numba compilation error

**Solution**: Ensure your custom function only uses numba-compatible operations:
- ✅ numpy operations
- ✅ basic math
- ✅ if/else logic
- ❌ Python objects
- ❌ list comprehensions
- ❌ dynamic types

### Issue: Reward not affecting behavior

**Check**:
1. Is the config actually being passed? Add print statement
2. Is the reward function returning varied values?
3. Try extreme rewards (0.0 or 1.0) to verify it's being used

## Migration from Old Code

### Old Code (Deprecated)
```python
# DON'T: Modify src/envs/n3il/rewards.py
from src.envs.n3il.rewards import get_value_nb

def my_custom_value(state, bound):
    # ...
```

### New Code (Recommended)
```python
# DO: Add to src/rewards.py
from src.rewards import register_value_function

@njit(cache=True, nogil=True)
@register_value_function('my_custom_value')
def my_custom_value(state, pts_upper_bound):
    # ...

# Then use in config
config = {'value_function': 'my_custom_value', ...}
```

## Summary

- ✅ **Single location** for all reward functions: `src/rewards.py`
- ✅ **Configurable** via `value_function` parameter
- ✅ **Extensible** via registration decorator
- ✅ **Type-safe** with numba compilation
- ✅ **No code duplication**

Happy experimenting! 🚀
