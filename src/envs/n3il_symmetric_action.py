import itertools
import numpy as np
print(np.__version__)
# np.random.seed(0)  # Removed global seed, will be set per experiment
from tqdm import trange
from numba import njit
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time
import datetime
import re
from collections import defaultdict
import pprint
import math
from typing import Tuple, List, Set, Callable, NamedTuple, Union, Optional, Iterable, Dict
from multiprocessing import Pool
from sympy import Rational, Integer
from sympy.core.numbers import igcd

import psutil
import os
import csv
import pandas as pd

from src.rewards.n3il_rewards import get_value_nb

from src.envs import N3il_with_symmetry

from src.utils.symmetry import get_d4_orbit

def get_d4_orbit(action, n, symmetry_mode):
    """
    Generate all valid (row, col) pairs corresponding to the 'symmetry_mode' string.
    The string is split by '_then_', and each part is treated as a generator applied
    to the current set of points.
    """
    row = action // n
    col = action % n
    points = {(row, col)}
    
    if not symmetry_mode:
        return points

    # Available transformations
    transforms = {
        'horizontal_flip':      lambda r, c: (r, n - 1 - c),          # Reflect across vertical axis |
        'vertical_flip':        lambda r, c: (n - 1 - r, c),          # Reflect across horizontal axis -
        'diagonal_flip':        lambda r, c: (c, r),                  # Reflect across main diagonal \
        'anti_diagonal_flip':   lambda r, c: (n - 1 - c, n - 1 - r),  # Reflect across anti-diagonal /
        'rotation_90':          lambda r, c: (c, n - 1 - r),          # 90 deg clockwise
        'rotation_180':         lambda r, c: (n - 1 - r, n - 1 - c),  # 180 deg
        'rotation_270':         lambda r, c: (n - 1 - c, r)           # 270 deg clockwise
    }
    
    # Parse operations sequence (e.g., "horizontal_flip_then_vertical_flip")
    operations = symmetry_mode.split('_then_')
    
    for op in operations:
        if op in transforms:
            func = transforms[op]
            # Apply this transform to all currently found points and extend the set
            new_points = set()
            for (r, c) in points:
                new_points.add(func(r, c))
            points.update(new_points)
            
    return points

class N3il_with_symmetry_and_symmetric_actions(N3il_with_symmetry):
    """
    N3il with both subgroup-based action filtering and symmetric action expansion.
    Symmetric action expansion generates all symmetric equivalents of a chosen action
    according to specified symmetry operations.
    """

    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        if 'symmetric_action' in args:
            self.symmetric_action_mode = args['symmetric_action']
        else:
            raise ValueError("symmetric_action parameter is required in args for N3il_with_symmetry_and_symmetric_actions.")


    def get_symmetric_actions(self, action):
        """
        Given an action (flattened index), return a set of all symmetric actions
        according to the specified symmetry_mode.
        """
        if not self.symmetric_action_mode:
            return {action}
        
        return {
            r * self.column_count + c
            for (r, c) in get_d4_orbit(action, self.row_count, self.symmetric_action_mode)
        }