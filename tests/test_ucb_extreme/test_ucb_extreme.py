"""
Test script for UCB-Extreme bandit policy implementation
Compare UCB1 vs UCB-Extreme on a simple test case
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs import N3il
from src.algos.mcts import MCTS

def test_ucb_policies():
    """Compare UCB1 and UCB-Extreme policies"""
    
    # Setup: Small grid for quick testing
    grid_size = 6
    game = N3il(
        row_count=grid_size, 
        column_count=grid_size,
        args={'display_state': False}
    )
    
    # Test configurations
    configs = [
        {
            'name': 'UCB1 (Default)',
            'args': {
                'num_searches': 100,
                'C': 1.4,
                'bandit_policy': 'ucb1',
                'process_bar': False
            }
        },
        {
            'name': 'UCB-Extreme',
            'args': {
                'num_searches': 100,
                'C': 1.4,
                'bandit_policy': 'ucb_extreme',
                'process_bar': False
            }
        }
    ]
    
    print("=" * 70)
    print("Testing UCB Policies on {}x{} Grid".format(grid_size, grid_size))
    print("=" * 70)
    
    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 70}")
        
        # Create MCTS instance
        mcts = MCTS(game, args=config['args'])
        
        # Initial state
        state = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Run one search from initial state
        print(f"Running {config['args']['num_searches']} MCTS searches...")
        action_probs = mcts.search(state)
        
        # Find best action
        best_action = np.argmax(action_probs)
        best_prob = action_probs[best_action]
        
        print(f"Best action: {best_action} (probability: {best_prob:.4f})")
        print(f"Policy: {config['args']['bandit_policy']}")
        print(f"Optimal value found: {mcts.optimal_value:.4f}")
        print(f"Number of optimal states: {len(mcts.optimal_terminal_states)}")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    test_ucb_policies()
