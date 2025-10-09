"""
MCTS (Monte Carlo Tree Search) Module

This module provides a modular implementation of MCTS and its variants for the
no-three-in-line problem. The code is organized into separate modules for easier
experimentation and ablation studies.

Main Components:
- utils.py: Utility functions including numba-compiled helpers, exploration decay, etc.
- simulation.py: Simulation strategies (random rollout, priority-based rollout)
- node.py: Node classes (basic Node, compressed Node, parallel Node variants)
- tree_search.py: Main MCTS algorithms (MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS)
- visualization.py: Tree visualization utilities
- evaluation.py: Evaluation and experiment running functions

Usage:
    from src.algos.mcts import MCTS, ParallelMCTS, evaluate

    # Run an experiment
    args = {
        'n': 10,
        'num_searches': 1000,
        'C': 1.4,
        'algorithm': 'MCTS',
        'environment': 'N3il',
        # ... other parameters
    }
    evaluate(args)
"""

import numpy as np
print(np.__version__)

# Import all core functionality from submodules
from src.algos.mcts.utils import (
    set_seeds,
    exploration_decay_nb,
    value_fn_nb,
    get_valid_moves_nb,
    get_valid_moves_subset_nb,
    check_collinear_nb,
    filter_top_priority_moves,
    select_outermost_with_tiebreaker,
    _pack_bits_bool2d,
    _unpack_bits_to_2d,
    _bit_clear_inplace,
    _bit_set_inplace
)

from src.algos.mcts.simulation import (
    simulate_nb,
    simulate_with_priority_nb,
    _rollout_many
)

from src.algos.mcts.node import (
    Node,
    Node_Compressed,
    LeafChildParallelNode
)

from src.algos.mcts.tree_search import (
    MCTS,
    ParallelMCTS,
    LeafChildParallelMCTS,
    MCGS,
    MCGSNode
)

from src.algos.mcts.visualization import (
    tree_visualization,
    save_final_visualization,
    save_comprehensive_html
)

from src.algos.mcts.evaluation import evaluate

# Export all public APIs
__all__ = [
    # Utility functions
    'set_seeds',
    'exploration_decay_nb',
    'value_fn_nb',
    'get_valid_moves_nb',
    'get_valid_moves_subset_nb',
    'check_collinear_nb',
    'filter_top_priority_moves',
    'select_outermost_with_tiebreaker',

    # Simulation functions
    'simulate_nb',
    'simulate_with_priority_nb',

    # Node classes
    'Node',
    'Node_Compressed',
    'LeafChildParallelNode',

    # MCTS algorithms
    'MCTS',
    'ParallelMCTS',
    'LeafChildParallelMCTS',
    'MCGS',
    'MCGSNode',

    # Visualization
    'tree_visualization',
    'save_final_visualization',
    'save_comprehensive_html',

    # Evaluation
    'evaluate'
]
