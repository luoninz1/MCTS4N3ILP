"""
MCTS Package for No-Three-In-Line Problem

This package provides a modular implementation of Monte Carlo Tree Search
algorithms for solving the no-three-in-line problem.
"""

from src.algos.mcts.tree_search import MCTS, ParallelMCTS, LeafChildParallelMCTS, MCGS
from src.algos.mcts.visualization import save_final_visualization, save_comprehensive_html

# Backward compatibility: import evaluate from new experiment module
# This allows existing code using "from src.algos.mcts import evaluate" to continue working
from src.experiment import evaluate

__all__ = [
    'MCTS',
    'ParallelMCTS',
    'LeafChildParallelMCTS',
    'MCGS',
    'evaluate',  # Now re-exported from src.experiment
    'save_final_visualization',
    'save_comprehensive_html'
]
