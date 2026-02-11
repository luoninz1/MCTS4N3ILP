import itertools
import numpy as np
# print(np.__version__)
# np.random.seed(0) 
from tqdm import trange
from numba import njit
import sys
import math
from typing import Tuple, List, Set, Callable, NamedTuple, Union, Optional, Iterable, Dict
from sympy import Rational, Integer
from sympy.core.numbers import igcd

from src.rewards.n3il_rewards import get_value_nb