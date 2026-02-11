from numba import njit
import numpy as np
from numpy.typing import NDArray

# Global configuration state
# [0]: Strategy ID 
#      0=Default/Reverse, 1=ExpGrowth, 2=Linear, 3=Gaussian, 4=Optimal3x3
# [1]: Generic Coefficient (e.g., -2.0 or 2.0)
# [2]: Target Ratio (e.g., 0.9 for Gaussian)
# [3]: Sigma Ratio (e.g., 0.05 for Gaussian)
REWARD_CONFIG = np.array([0.0, -2.0, 0.9, 0.05], dtype=np.float64)

def set_reward_strategy(name: str):
    """
    Python-side helper to configure the reward strategy.
    Call this BEFORE starting MCTS.
    """
    if name == "default" or name == "exp_reverse":
        # Reverse Reward (Default active in your code)
        REWARD_CONFIG[:] = [0.0, -2.0, 0.0, 0.0]
    elif name == "exp_growth":
        # Exponential Growth (Value for FVAS)
        REWARD_CONFIG[:] = [1.0, 2.0, 0.0, 0.0]
    elif name == "linear":
        # Simple Linear Inverse
        REWARD_CONFIG[:] = [2.0, 0.0, 0.0, 0.0]
    elif name == "gaussian":
        # Gaussian peak at target 0.9n
        REWARD_CONFIG[:] = [3.0, 0.0, 0.9, 0.05]
    elif name == "optimal_3x3":
        # Optimal for 3x3 Minimal Complete Set
        REWARD_CONFIG[:] = [4.0, 0.0, 0.0, 0.0]
    else:
        print(f"Warning: Unknown reward strategy '{name}', using default.")
        REWARD_CONFIG[:] = [0.0, -2.0, 0.0, 0.0]

## Remember Also Adjust get_value_nb in collinear_for_mcts.py !!!!!!!!!!!!!!!!!!!!!!!!!!!!
@njit(cache=True, nogil=True)
def get_value_exp_norm_nb(state: NDArray, pts_upper_bound: int, coeff: float = 2.0):
    total = np.sum(state)
    n = pts_upper_bound/2
    return np.exp(coeff * ((total-n) / n))

@njit(cache=True, nogil=True)
def get_value_for_FVAS_nb(state, pts_upper_bound):
    total = np.sum(state) - 1
    n = pts_upper_bound/2
    return np.exp(2.0 * ((total-n) / n))

## Default rewarding function
## ACTS AS DISPATCHER BASED ON GLOBAL CONFIG
@njit(cache=True, nogil=True)
def get_value_nb(state: NDArray, pts_upper_bound: int):
    # Retrieve configuration from global array (safe in Numba)
    mode = REWARD_CONFIG[0]
    
    # Common calculations
    # Use float64 for precision in math calcs
    total_f = np.float64(np.sum(state))
    n_f = np.float64(pts_upper_bound) / 2.0
    
    # --- 0. Default: Reverse Reward (-2.0) ---
    if mode == 0.0:
        coeff = REWARD_CONFIG[1] # -2.0
        return np.exp(coeff * ((total_f - n_f) / n_f))
        
    # --- 1. Exponential Growth (2.0) ---
    elif mode == 1.0:
        coeff = REWARD_CONFIG[1] # 2.0
        return np.exp(coeff * ((total_f - n_f) / n_f))
        
    # --- 2. Linear Inverse ---
    elif mode == 2.0:
        # (1.2*n - total) / n
        return (1.2 * n_f - total_f) / n_f
        
    # --- 3. Gaussian Peak (Target based) ---
    elif mode == 3.0:
        target_ratio = REWARD_CONFIG[2] # 0.9
        sigma_ratio = REWARD_CONFIG[3]  # 0.05
        
        target = target_ratio * n_f
        sigma = np.maximum(sigma_ratio * n_f, 1e-12)
        
        return np.exp(-0.5 * ((total_f - target) / sigma) ** 2)

    # --- 4. Optimal for 3x3 ---
    elif mode == 4.0:
        # (1.6*n - total) * n / (1.6 - 1.3)
        return (1.6 * n_f - total_f) * n_f / 0.3 # (1.6 - 1.3)
        
    # Fallback to default if something goes wrong
    return np.exp(-2.0 * ((total_f - n_f) / n_f))