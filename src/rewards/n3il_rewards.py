from numba import njit
import numpy as np
from numpy.typing import NDArray

## Remember Also Adjust get_value_nb in collinear_for_mcts.py !!!!!!!!!!!!!!!!!!!!!!!!!!!!
@njit(cache=True, nogil=True)
def get_value_exp_norm_nb(state: NDArray, pts_upper_bound: int, coeff: float = 2.0):
    total = np.sum(state)
    n = pts_upper_bound/2
    
    # 2. Exponential Growth/Decay (Strong preference for fewer points)
    return np.exp(coeff * ((total-n) / n))  # Range: [e^-2, 1] ≈ [0.135, 1]


## Default rewarding function
## Remember Also Adjust get_value_nb in collinear_for_mcts.py !!!!!!!!!!!!!!!!!!!!!!!!!!!!
@njit(cache=True, nogil=True)
def get_value_nb(state: NDArray, pts_upper_bound: int):
    total = np.sum(state)
    n = pts_upper_bound/2
    
    # === REVERSE REWARDING FUNCTIONS (prefer smaller point counts) ===
    
    # 1. Simple Linear Inverse: 1.0 for empty board, 0.0 for full board
    # return (1.2*n - total) / n  # Range: [0, 1]
    
    # 2. Exponential Decay (Strong preference for fewer points)
    return np.exp(2.0 * ((total-n) / n))  # Range: [e^-2, 1] ≈ [0.135, 1]
    # return np.exp(-1.0 * (total / n))  # Range: [e^-1, 1] ≈ [0.368, 1]
    # return np.exp(-0.5 * (total / n))  # Range: [e^-0.5, 1] ≈ [0.607, 1]
    
    # 3. Power Functions (Adjustable curvature)
    # return ((n - total) / n) ** 2  # Quadratic preference: [0, 1]
    # return ((n - total) / n) ** 0.5  # Square root preference: [0, 1]
    # return ((n - total) / n) ** 3  # Cubic preference (very aggressive): [0, 1]
    
    # 4. Sigmoid-based (Smooth transition around target)
    # target = n * 0.3  # Target 30% of grid filled
    # return 1.0 / (1.0 + np.exp(0.5 * (total - target)))  # Range: ≈[0, 1]
    # return 1.0 / (1.0 + np.exp(1.0 * (total - target)))  # Steeper transition
    
    # 5. Logarithmic Penalty
    # return max(0, 1.0 - np.log(1.0 + total) / np.log(1.0 + n))  # Range: [0, 1]
    
    # 6. ReLU-based with different thresholds
    # return max(0, (1.2 * n - total) / n)  # Reward up to 120% of n: [0, 1.2]
    # return max(0, (1.5 * n - total) / n)  # Current: reward up to 150% of n
    
    # === OPTIMAL FOR 3x3 MINIMAL COMPLETE SET (4 points) ===
    # Simple linear inverse works best for finding exact minimal sets
    # return (1.6*n - total) * n  / (1.6 - 1.3) # Range: [0, 1], 1.0 for empty, 0.0 for full !!!CURRENT OPTIMAL!!!

    # Baseline rewarding function
    '''
    baseline = 1.6 * n
    theoretical_min = 1.3 * n
    num = baseline - total
    if num > 0:
        return num / (baseline - theoretical_min)  # Range: [0, 1], 1.0 for empty, 0.0 for full
    if num <= 0:
        return num / (baseline - theoretical_min)  # Range: [-1, 0], 0.0 for empty, -1.0 for full
    '''
    # Numba-safe scalar casts
    total = np.float64(np.sum(state))
    n = np.float64(pts_upper_bound) / 2.0

    # Target and normalization
    target = 0.9 * n
    max_possible = 2.0 * n
    eps = np.float64(1e-12)
    span = np.maximum(max_possible - target, eps)  # avoid division by zero
    # Normalized distance: 0 at target, 1 at 2n (can be < 0 if total < target)
    tnorm = (total - target) / span

    # ---- Choose ONE of the following returns (uncomment exactly one) ----

    # 2) Quadratic (penalizes farther from target more strongly)
    # return np.clip(1.0 - tnorm * tnorm, 0.0, 1.0)

    # 3) Gaussian peak at target (default active; sharp pull to 0.9n)
    # sigma = np.maximum(0.05 * n, eps)  # controls sharpness
    # return np.exp(-0.5 * ((total - target) / sigma) ** 2)

    # 4) Logistic decay from target upward
    # k = 6.0 / np.maximum(n, 1.0)
    # return 1.0 / (1.0 + np.exp(k * (total - target)))

    # 5) Rational distance penalty (gentler tail)
    # alpha = 2.0 / np.maximum(n, 1.0)
    # return 1.0 / (1.0 + alpha * np.abs(total - target))

    # 6) Piecewise: full at/below target, then linear drop to 0 at 2n
    # if total <= target:
    #     return 1.0
    # else:
    #     return np.maximum(0.0, 1.0 - (total - target) / span)

    # 7) Cosine half-wave on [target, 2n] (smooth with zero slope at target)
    # x = np.clip(tnorm, 0.0, 1.0)               # map [target,2n] -> [0,1]
    # return 0.5 * (1.0 + np.cos(np.pi * x))     # 1 at target, 0 at 2n

    # ------------ Positive-direction variants (optimum at 2n) ------------
    # Use these if you want to test the opposite objective (larger total better).
    # 1+) Linear increasing from target to 2n
    # return np.clip(tnorm, 0.0, 1.0)

    # 2+) Quadratic increasing (slow start, faster near 2n)
    # x = np.clip(tnorm, 0.0, 1.0)
    # return x * x

    # 3+) Exponential rise (very low until near 2n)
    # x = np.clip(tnorm, 0.0, 1.0)
    # k = 4.0
    # return (np.exp(k * x) - 1.0) / (np.exp(k) - 1.0)