from numba import njit
import numpy as np

# Global configuration state
# [0]: Strategy ID 
#      0=No Decay (always explore), 1=Sqrt Decay, 2=Linear Decay
EXPLORATION_DECAY_CONFIG = np.array([0.0], dtype=np.float64)

def set_exploration_decay_strategy(name: str):
    """
    Python-side helper to configure the exploration decay strategy.
    Call this BEFORE starting MCTS.
    
    Args:
        name: Strategy name, one of:
            - "no_decay": Always return 1 (constant exploration)
            - "sqrt_decay": Gentle decay using sqrt(x), returns 1 - 0.85 * sqrt(x)
            - "linear_decay": Linear decay, returns 1 - x
    """
    if name == "no_decay":
        EXPLORATION_DECAY_CONFIG[0] = 0.0
    elif name == "sqrt_decay":
        EXPLORATION_DECAY_CONFIG[0] = 1.0
    elif name == "linear_decay":
        EXPLORATION_DECAY_CONFIG[0] = 2.0
    else:
        print(f"Warning: Unknown exploration decay strategy '{name}', using 'no_decay'.")
        EXPLORATION_DECAY_CONFIG[0] = 0.0

@njit(cache=True, nogil=True)
def exploration_decay_nb(x):
    """
    Exploration decay function from (0,1) to (1,0).
    Monotone decreasing based on search progress.
    
    Args:
        x: Search progress ratio in [0, 1]
        
    Returns:
        Decay multiplier (typically starts near 1.0 at x=0)
    """
    mode = EXPLORATION_DECAY_CONFIG[0]
    
    # --- 0. No Decay (Always explore) ---
    if mode == 0.0:
        return 1.0
        
    # --- 1. Square Root Decay ---
    elif mode == 1.0:
        return 1.0 - 0.85 * np.sqrt(x)
        
    # --- 2. Linear Decay ---
    elif mode == 2.0:
        return 1.0 - x
        
    # Fallback to no decay
    return 1.0
