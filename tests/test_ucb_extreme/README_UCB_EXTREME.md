# UCB-Extreme Implementation

## Overview

UCB-Extreme is a bandit allocation strategy that focuses on discovering the **maximum reward** rather than maximizing expected cumulative reward. This is particularly useful for problems where finding the best single solution is more important than average performance.

## Implementation Details

### Key Changes

1. **Node Initialization** (`Node.__init__` and `Node_Compressed.__init__`)
   - Adds `_policy` attribute from `args['bandit_policy']`
   - For `ucb_extreme`: initializes `q_max = -np.inf`
   - For `ucb1` (default): initializes `value_sum = 0`

2. **UCB Calculation** (`get_ucb`)
   - `ucb_extreme`: uses `q_max` as the exploitation term
   - `ucb1`: uses `value_sum / visit_count` as the exploitation term
   - Exploration term remains unchanged: `C * sqrt(log(N) / n_i)`

3. **Backpropagation** (`backpropagate`)
   - `ucb_extreme`: `q_max = max(q_max, value)`
   - `ucb1`: `value_sum += value`

4. **Virtual Loss**
   - `ucb_extreme`: Only modifies `visit_count` (not `q_max`)
   - `ucb1`: Modifies both `value_sum` and `visit_count`
   - Rationale: Virtual loss should only increase exploration penalty, not affect the maximum observed value

5. **Visualization** (`_get_node_label`)
   - Displays appropriate statistics based on policy:
     - `ucb_extreme`: Shows `Q_max`
     - `ucb1`: Shows `Value Sum` and `Avg Value`

## Usage

### Basic Usage

```python
from src.algos.mcts import MCTS
from src.envs import N3il

# Create game environment
game = N3il(row_count=6, column_count=6)

# UCB1 (Default behavior)
mcts_ucb1 = MCTS(game, args={
    'num_searches': 1000,
    'C': 1.4,
    'bandit_policy': 'ucb1'  # Can be omitted (default)
})

# UCB-Extreme
mcts_extreme = MCTS(game, args={
    'num_searches': 1000,
    'C': 1.4,
    'bandit_policy': 'ucb_extreme'  # NEW: Use extreme bandit policy
})

# Run search
state = np.zeros((6, 6), dtype=np.uint8)
action_probs = mcts_extreme.search(state)
```

### With Node Compression

```python
# UCB-Extreme with compressed nodes
mcts = MCTS(game, args={
    'num_searches': 1000,
    'C': 1.4,
    'bandit_policy': 'ucb_extreme',
    'node_compression': True  # Works with both policies
})
```

### With Tree Visualization

```python
# Visualize the tree to compare policies
mcts = MCTS(game, args={
    'num_searches': 100,
    'C': 1.4,
    'bandit_policy': 'ucb_extreme',
    'tree_visualization': True,
    'pause_at_each_step': False
})

# Node labels will show Q_max instead of Avg Value
action_probs = mcts.search(state)

# Save visualization
MCTS.save_final_visualization(
    web_viz_dir='./figures',
    experiment_name='ucb_extreme_test'
)
```

## Theory: UCB-Extreme vs UCB1

### UCB1 Formula
$$UCB_1(i) = \bar{X}_i + C\sqrt{\frac{\ln N}{n_i}}$$

Where:
- $\bar{X}_i$ = average reward from arm $i$
- $N$ = total number of selections
- $n_i$ = number of times arm $i$ was selected

**Goal**: Maximize expected cumulative reward

### UCB-Extreme Formula
$$UCB_{extreme}(i) = \hat{Q}_i + C\sqrt{\frac{\ln N}{n_i}}$$

Where:
- $\hat{Q}_i$ = **maximum** observed reward from arm $i$
- Exploration term unchanged

**Goal**: Discover the arm with maximum reward

### When to Use Each

**Use UCB1 when:**
- You care about average performance
- Optimizing cumulative reward
- Standard multi-armed bandit problems

**Use UCB-Extreme when:**
- Finding the single best solution is critical
- Symbolic regression, expression discovery
- Combinatorial optimization (like no-three-in-line problem)
- Quality diversity search

## Testing

Run the test script:

```bash
cd tests/test_ucb_extreme
python test_ucb_extreme.py
```

## Experimental Validation

To compare policies systematically:

1. **Small-scale experiments**: Use `test_ucb_extreme.py`
2. **Ablation study**: Modify `tests/ablation_study/ablation.py` to include `bandit_policy` as a parameter
3. **Visualization**: Enable tree visualization to see how node values differ

## Notes

- Both policies are compatible with all existing features (compression, virtual loss, symmetry handling)
- The exploration term formula remains identical
- Node visit counts are tracked the same way
- Default behavior (`ucb1`) is unchanged for backward compatibility

## References

[1] Streeter, M., & Smith, S. F. (2006). "A Simple Distribution-Free Approach to the Max k-Armed Bandit Problem"
[2] Original paper from the attached PDF on symbolic regression

## Future Work

Consider implementing:
1. **Action selection**: Currently selects by visit count; could select by `q_max` for UCB-Extreme
2. **Hybrid policies**: Adaptive switching between UCB1 and UCB-Extreme
3. **Temperature-based exploration**: Soften the max operator during early search
