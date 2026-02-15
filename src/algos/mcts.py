import itertools
import numpy as np
print(np.__version__)
# np.random.seed(0)  # Removed global seed, will be set per experiment
from tqdm import trange
from numba import njit
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, as_completed
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
from src.envs import N3il, N3il_with_symmetry, N3il_with_symmetry_and_symmetric_actions, supnorm_priority, supnorm_priority_array, N4il
from src.utils.symmetry import get_d4_orbit

import io
import base64
from pyvis.network import Network

import psutil
import os
import warnings
import uuid

from src.rewards.n3il_rewards import get_value_nb
from src.utils.exploration_decay import exploration_decay_nb

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)
        
        # Bandit policy: 'ucb1' (default) or 'ucb_extreme'
        self._policy = args.get('bandit_policy', 'ucb1')
        if self._policy == 'ucb_extreme':
            self.q_max = -np.inf  # Track maximum observed value
        else:
            self.value_sum = 0    # Track sum of values (for average)

        if parent is None:
            self.level = np.sum(state)  # Level is the number of points placed
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_with_symmetry')):
                self.action_space = game.get_valid_moves(state)
                self.valid_moves = game.filter_valid_moves_by_symmetry(
                    self.action_space, state
                ).copy()
            else:
                self.valid_moves = game.get_valid_moves(state)
                self.action_space = self.valid_moves.copy()
        else:
            self.level = parent.level + 1
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_subset_with_symmetry')):
                self.action_space = game.get_valid_moves_subset(
                    parent.state, parent.action_space, self.action_taken, current_state=state)
                self.valid_moves = game.filter_valid_moves_by_symmetry(
                    self.action_space, state
                ).copy()
            else:
                self.valid_moves = game.get_valid_moves_subset(
                    parent.state, parent.action_space, self.action_taken, current_state=state)
                self.action_space = self.valid_moves.copy()
        
        # Ensure action_space is immutable
        self.action_space.flags.writeable = False

        self.is_full = False
        self._cached_ucb = None     # Cached UCB value
        self._ucb_dirty = True      # Indicates whether the cached UCB is stale

    def apply_virtual_loss(self):
        with self.lock:
            if self._policy == 'ucb1':
                self.value_sum -= self._vl
            # For ucb_extreme: don't modify q_max, only visit_count affects exploration
            self.visit_count += 1
            self._ucb_dirty = True  # Mark UCB as outdated

    def revert_virtual_loss(self):
        with self.lock:
            if self._policy == 'ucb1':
                self.value_sum += self._vl
            # For ucb_extreme: q_max was not modified, nothing to revert
            self._ucb_dirty = True  # Mark UCB as outdated

    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0

    def select(self, exploration_factor):
        best_child = None
        best_ucb = -np.inf
        log_N = math.log(self.visit_count)

        for child in self.children:
            ucb = self.get_ucb(child, exploration_factor, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, exploration_factor, log_N=None):
        if log_N is None:
            log_N = math.log(self.visit_count)

        # Optimization: Optimistic read without lock
        if not child._ucb_dirty and child._cached_ucb is not None:
            return child._cached_ucb

        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb

            # Calculate q_value based on bandit policy
            if child._policy == 'ucb_extreme':
                q_value = child.q_max  # Use maximum observed reward
            else:
                q_value = child.value_sum / child.visit_count  # Use average reward
            
            exploration_value = exploration_factor * math.sqrt(log_N / child.visit_count)
            ucb = q_value + exploration_value
            # print("Exploit:", q_value)
            # print("Explore:", exploration_value)
            child._cached_ucb = ucb
            child._ucb_dirty = False
            return ucb

    def expand(self):
        valid_indices = np.where(self.valid_moves == 1)[0]
        action = np.random.choice(valid_indices)
        self.valid_moves[action] = 0

        if np.sum(self.valid_moves) == 0:
            self.is_full = True

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, action_space_of_state=self.action_space)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        tmp = self.state.copy()
        value, terminal_state = self.game.simulate(tmp)
        return value, terminal_state

    def backpropagate(self, value):
        with self.lock:
            if self._policy == 'ucb_extreme':
                self.q_max = max(self.q_max, value)  # Update maximum observed value
            else:
                self.value_sum += value  # Accumulate for average
            self._ucb_dirty = True  # Mark UCB as outdated
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

# ========= Bit-pack utilities =========
def _pack_bits_bool2d(arr2d: np.ndarray) -> np.ndarray:
    """
    Pack a 2D 0/1 or bool array into a 1D uint8 bit vector using bitorder='big'.
    """
    # Ensure uint8 0/1
    a = arr2d.astype(np.uint8, copy=False)
    return np.packbits(a.reshape(-1), bitorder='big')

def _unpack_bits_to_2d(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Unpack a 1D uint8 bit vector to a 2D uint8 array (0/1) with given shape.
    """
    flat = np.unpackbits(bits, bitorder='big')
    need = rows * cols
    if flat.size > need:
        flat = flat[:need]
    return flat.reshape((rows, cols)).astype(np.uint8, copy=False)

def _bit_clear_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Clear (set to 0) the bit at flat index idx in the packed array (bitorder='big').
    Uses a non-negative mask to avoid OverflowError from bitwise NOT on Python ints.
    """
    byte_i = idx // 8
    off    = idx % 8
    # Build a clear mask: 0xFF with target bit cleared
    clear_mask = np.uint8(0xFF ^ (1 << (7 - off)))
    bits[byte_i] &= clear_mask

def _bit_set_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Set (to 1) the bit at flat index idx in the packed array (bitorder='big').
    """
    byte_i = idx // 8
    off    = idx % 8
    bits[byte_i] |= np.uint8(1 << (7 - off))


class Node_Compressed:
    """
    Drop-in compatible Node that implements Scheme B:
    - Store state / valid_moves / action_space as bit-packed arrays (1 bit per cell).
    - Provide properties .state, .valid_moves, .action_space to return unpacked views (uint8 0/1).
    - Internal methods operate on packed bits to reduce memory and allocations.
    Notes:
      * Compatibility: external code that reads node.state / node.valid_moves continues to work.
      * External code that mutates node.valid_moves should not be relied upon (same as original),
        but we return an array copy for safety.
    """
    # Keep the same public attributes (exposed via properties where needed)
    __slots__ = (
        'game','args','parent','action_taken',
        'children','visit_count','value_sum','lock','_vl',
        'level','is_full','_cached_ucb','_ucb_dirty',
        # packed payloads
        '_rows','_cols','_state_bits','_valid_bits','_action_bits', '_policy','q_max'
    )

    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)
        
        # Bandit policy: 'ucb1' (default) or 'ucb_extreme'
        self._policy = args.get('bandit_policy', 'ucb1')
        if self._policy == 'ucb_extreme':
            self.q_max = -np.inf  # Track maximum observed value
        else:
            self.value_sum = 0.0  # Track sum of values (for average)

        # grid shape
        self._rows = getattr(game, 'row_count', state.shape[0])
        self._cols = getattr(game, 'column_count', state.shape[1] if state.ndim > 1 else self._rows)

        # --- pack state ---
        # Expect state as 2D 0/1 (uint8 or bool)
        self._state_bits = _pack_bits_bool2d(state)

        # --- compute valid/action masks using the same game API as original ---
        if parent is None:
            # level = number of points placed
            self.level = int(np.sum(state))
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_with_symmetry')):
                action_space = game.get_valid_moves(state)
                valid_moves  = game.filter_valid_moves_by_symmetry(action_space, state).copy()
            else:
                valid_moves  = game.get_valid_moves(state)
                action_space = valid_moves.copy()
        else:
            self.level = parent.level + 1
            # For subset calls, pass parent's action_space (unpacked) and parent.state (unpacked)
            parent_state = parent.state
            parent_action_space = parent.action_space
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_subset_with_symmetry')):
                action_space = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken, current_state=state)
                valid_moves  = game.filter_valid_moves_by_symmetry(action_space, self.state,).copy()  # self.state property unpacks
            else:
                valid_moves  = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken, current_state=state)
                action_space = valid_moves.copy()

        # --- pack masks & discard large arrays ---
        self._valid_bits  = _pack_bits_bool2d(valid_moves.reshape(self._rows, self._cols))
        self._action_bits = _pack_bits_bool2d(action_space.reshape(self._rows, self._cols))

        self.is_full = False
        self._cached_ucb = None
        self._ucb_dirty = True

    # ---------- compatibility properties ----------
    @property
    def state(self) -> np.ndarray:
        # Return a fresh 2D uint8 (0/1) array
        return _unpack_bits_to_2d(self._state_bits, self._rows, self._cols)

    @property
    def valid_moves(self) -> np.ndarray:
        # Return a fresh 1D uint8 (0/1) vector, write-protected to mimic immutability contract
        vm = _unpack_bits_to_2d(self._valid_bits, self._rows, self._cols).reshape(-1)
        vm.flags.writeable = False
        return vm

    @property
    def action_space(self) -> np.ndarray:
        am = _unpack_bits_to_2d(self._action_bits, self._rows, self._cols).reshape(-1)
        am.flags.writeable = False
        return am

    # ---------- drop-in methods (logic aligned with original) ----------
    def apply_virtual_loss(self):
        with self.lock:
            if self._policy == 'ucb1':
                self.value_sum -= self._vl
            # For ucb_extreme: don't modify q_max, only visit_count affects exploration
            self.visit_count += 1
            self._ucb_dirty = True

    def revert_virtual_loss(self):
        with self.lock:
            if self._policy == 'ucb1':
                self.value_sum += self._vl
            # For ucb_extreme: q_max was not modified, nothing to revert
            self._ucb_dirty = True

    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0

    def select(self, exploration_factor):
        best_child = None
        best_ucb = -np.inf
        # avoid log(0)
        log_N = math.log(self.visit_count) if self.visit_count > 0 else 0.0

        for child in self.children:
            ucb = self.get_ucb(child, exploration_factor, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, exploration_factor, log_N=None):
        if log_N is None:
            log_N = math.log(self.visit_count) if self.visit_count > 0 else 0.0

        # Optimization: Optimistic read without lock
        if not child._ucb_dirty and child._cached_ucb is not None:
            return child._cached_ucb

        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb

            # Calculate q_value based on bandit policy
            if child._policy == 'ucb_extreme':
                q_value = child.q_max  # Use maximum observed reward
            else:
                q_value = child.value_sum / max(1, child.visit_count)  # Use average reward
            
            exploration_value = exploration_factor * math.sqrt(max(1e-12, log_N) / max(1, child.visit_count))
            ucb = q_value + exploration_value
            child._cached_ucb = ucb
            child._ucb_dirty = False
            return ucb

    def _valid_sum(self) -> int:
        # Fast count of set bits
        flat = np.unpackbits(self._valid_bits, bitorder='big')
        return int(flat[: self._rows * self._cols].sum())

    def expand(self):
        # Choose a random valid action; work on packed bits to avoid storing big arrays
        flat_valid = np.unpackbits(self._valid_bits, bitorder='big')[: self._rows * self._cols]
        valid_indices = np.flatnonzero(flat_valid)
        if valid_indices.size == 0:
            self.is_full = True
            return None

        action = int(np.random.choice(valid_indices))
        # consume this action (clear its bit)
        _bit_clear_inplace(self._valid_bits, action)

        # mark is_full if no moves remain
        if flat_valid.sum() - 1 == 0:
            self.is_full = True

        # Build child state as in original (copy, then get_next_state)
        child_state = self.state.copy()  # property: unpack current state
        child_state = self.game.get_next_state(child_state, action, action_space_of_state=self.action_space)

        # Create child node (compressed)
        child = Node_Compressed(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        # Unpack to 2D array; simulation mutates a copy
        tmp = self.state.copy()
        if self.args.get("simulate_with_priority", False):
            # priority is no longer supported in this version.
            raise NotImplementedError("simulate_with_priority is not supported in Node_Compressed version.")
            value, terminal_state = simulate_with_priority_nb(tmp, 
                                            self.game.row_count,
                                            self.game.column_count,
                                            self.game.pts_upper_bound,
                                            self.game.priority_grid,
                                            self.args['TopN'])
            return value, terminal_state
        else:
            value, terminal_state = self.game.simulate(tmp)
            return value, terminal_state

    def backpropagate(self, value):
        with self.lock:
            if self._policy == 'ucb_extreme':
                self.q_max = max(self.q_max, value)  # Update maximum observed value
            else:
                self.value_sum += value  # Accumulate for average
            self._ucb_dirty = True
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    # Global storage for all trials and steps (class variable)
    global_trial_data = []
    
    def __init__(self, game, args={
        'num_searches': 1000,
        'C': 1.4
    }):
        self.game = game
        self.args = args
        self.snapshots = []  # Store tree snapshots for multi-snapshot viewing
        self.trial_id = None  # Will be set when starting a new trial
        
        # Storage for optimal terminal states
        self.optimal_terminal_states = [] 
        self.optimal_value = -float('inf')
        self.optimal_states_lock = threading.Lock()

    def update_optimal_states(self, value, state):
        """
        Thread-safe update of optimal terminal states.
        """
        # If tracking is not enabled, do nothing (or do it always? user said enabled by args 'save_optimal_terminal_state')
        # Checking logic inside here or caller?
        # Caller does not check args easily in Node.simulate, but MCTS instance has args.
        if not self.args.get('save_optimal_terminal_state', False):
            return

        with self.optimal_states_lock:
            if value > self.optimal_value:
                self.optimal_value = value
                self.optimal_terminal_states = [{
                    'state': state.copy(),
                    'timestamp': time.time(),
                    'id': uuid.uuid4().hex
                }]
            elif value == self.optimal_value and self.args.get('save_all_optimal_terminal_states', False):
                # Check for duplicates by exact state match
                is_new = True
                for existing in self.optimal_terminal_states:
                    if np.array_equal(existing['state'], state):
                        is_new = False
                        break
                if is_new:
                    self.optimal_terminal_states.append({
                        'state': state.copy(),
                        'timestamp': time.time(),
                        'id': uuid.uuid4().hex
                    })

    def _state_to_image_base64(self, state):
        """
        Convert a game state to a base64-encoded image using the game's display_state method.
        """
        # Temporarily modify the game's display settings to avoid saving files
        original_display = self.game.args.get('display_state', False)
        original_figure_dir = self.game.args.get('figure_dir', '')
        
        # Create a temporary figure
        plt.figure(figsize=(4, 4))
        
        # Get the grid info
        rows, cols = self.game.row_count, self.game.column_count
        
        # Plot the state (simplified version of display_state)
        y_idx, x_idx = np.nonzero(state)
        y_disp = rows - 1 - y_idx
        plt.scatter(x_idx, y_disp, s=200, c='blue', linewidths=0.5)
        
        # Draw grid
        plt.xticks(range(cols))
        plt.yticks(range(rows))
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.gca().set_aspect('equal')
        
        # Remove axes labels and ticks for cleaner look
        plt.xticks([])
        plt.yticks([])
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, pad_inches=0.1)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"

    def _get_node_label(self, node, iter_num=None):
        """
        Generate a label for a node showing its statistics.
        """
        # Display different stats based on bandit policy
        policy = getattr(node, '_policy', 'ucb1')
        
        if policy == 'ucb_extreme':
            # For UCB-Extreme: show q_max instead of average
            q_value = getattr(node, 'q_max', -np.inf)
            label = f"Visits: {node.visit_count}\n"
            label += f"Q_max: {q_value:.3f}\n"
        else:
            # For UCB1: show value_sum and average
            value_sum = getattr(node, 'value_sum', 0.0)
            avg_value = value_sum / node.visit_count if node.visit_count > 0 else 0
            label = f"Visits: {node.visit_count}\n"
            label += f"Value Sum: {value_sum:.3f}\n"
            label += f"Avg Value: {avg_value:.3f}\n"
        
        # Calculate UCB if this node has a parent
        ucb = 0
        if node.parent is not None and node.visit_count > 0:
            try:
                ucb = node.parent.get_ucb(node, iter_num or 0)
            except:
                ucb = 0
        
        label += f"UCB: {ucb:.3f}"
        
        return label

    def tree_visualization(self, root, snapshot_name="MCTS Tree"):
        """
        Create a tree visualization using pyvis and save it as a snapshot.
        Ensures all expanded nodes are captured, including nodes with no children.
        """
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        net.barnes_hut()
        
        # Collect all nodes using DFS to ensure we capture everything
        all_nodes = []
        visited = set()
        
        def collect_nodes_dfs(node, level=0):
            if id(node) in visited:
                return
            visited.add(id(node))
            all_nodes.append((node, level))
            
            # Recursively collect children
            for child in node.children:
                collect_nodes_dfs(child, level + 1)
        
        # Start DFS from root
        collect_nodes_dfs(root)
        
        print(f"Tree visualization: Found {len(all_nodes)} nodes total")
        
        # Prepare JSON-serializable data for nodes and edges
        json_nodes = []
        json_edges = []
        node_mapping = {}
        
        # Add nodes to the network
        for i, (node, level) in enumerate(all_nodes):
            # Generate unique node ID
            current_id = f"node_{i}"
            node_mapping[id(node)] = current_id
            
            # Get node image and label
            img_base64 = self._state_to_image_base64(node.state)
            label = self._get_node_label(node)
            
            # Determine node color based on properties
            color = "#4CAF50"  # Default green
            if node.is_fully_expanded():
                color = "#2196F3"  # Blue for fully expanded
            elif len(node.children) == 0 and not node.is_fully_expanded():
                color = "#FF9800"  # Orange for leaf nodes that could expand
            elif np.sum(node.valid_moves) == 0:
                color = "#F44336"  # Red for terminal nodes
            
            # Add node to pyvis network
            net.add_node(
                current_id,
                label=label,
                image=img_base64,
                shape="image",
                size=30,
                level=level,
                color=color,
                title=f"Action: {node.action_taken}\n{label}\nChildren: {len(node.children)}\nValid moves left: {np.sum(node.valid_moves)}"
            )
            
            # Prepare JSON data for this node - split label into lines for proper display
            label_lines = label.split('\n')
            # Create title with proper newline escaping
            escaped_label = label.replace('\n', '\\n')
            title_text = f"Action: {node.action_taken}\\n{escaped_label}\\nChildren: {len(node.children)}\\nValid moves left: {np.sum(node.valid_moves)}"
            
            json_nodes.append({
                "id": current_id,
                "label": label_lines,  # Use array of lines instead of single string
                "image": img_base64,
                "shape": "image", 
                "size": 30,
                "level": level,
                "color": color,
                "title": title_text,
                "x": i * 100,  # Simple layout
                "y": level * 150
            })
        
        # Add edges between nodes
        for node, _ in all_nodes:
            current_id = node_mapping[id(node)]
            for child in node.children:
                if id(child) in node_mapping:  # Ensure child was also collected
                    child_id = node_mapping[id(child)]
                    net.add_edge(current_id, child_id)
                    json_edges.append({
                        "from": current_id,
                        "to": child_id,
                        "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}
                    })
        
        print(f"Tree visualization: Added {len(net.nodes)} nodes and {len(net.edges)} edges")
        
        # Configure layout
        net.set_options("""
        var options = {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "shakeTowards": "roots",
                    "levelSeparation": 150,
                    "nodeSpacing": 100
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 100,
                    "springConstant": 0.01,
                    "nodeDistance": 120,
                    "damping": 0.09
                },
                "maxVelocity": 50,
                "solver": "hierarchicalRepulsion",
                "stabilization": {"iterations": 100}
            },
            "nodes": {
                "font": {
                    "size": 12,
                    "color": "white"
                }
            },
            "edges": {
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "vertical",
                    "roundness": 0.4
                }
            }
        }
        """)
        
        # Store this snapshot with trial information
        snapshot_data = {
            'name': snapshot_name,
            'network': net,
            'html': net.generate_html(),
            'trial_id': getattr(self, 'trial_id', 'unknown'),
            'step_number': len(self.snapshots),
            'total_nodes': len(all_nodes),
            'args': self.args.copy(),  # Store configuration for reference
            'json_nodes': json_nodes,  # Add JSON data for better HTML generation
            'json_edges': json_edges
        }
        
        self.snapshots.append(snapshot_data)
        
        # Also add to global trial data for multi-trial viewing
        MCTS.global_trial_data.append(snapshot_data)
        
        return net

    @classmethod
    def clear_global_data(cls):
        """Clear all global trial data. Call this at the start of a new experiment set."""
        cls.global_trial_data.clear()
        print("Global trial data cleared.")
    
    @classmethod 
    def save_final_visualization(cls, web_viz_dir=None, experiment_name="mcts_experiment"):
        """
        Save the final comprehensive visualization at the end of all trials.
        """
        if not cls.global_trial_data:
            print("No global trial data to save.")
            return None
            
        if web_viz_dir is None:
            web_viz_dir = './web_visualization'
        
        # Create the web visualization directory
        os.makedirs(web_viz_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(web_viz_dir, f"{experiment_name}_comprehensive_{timestamp}.html")
        
        # Save comprehensive visualization
        cls.save_comprehensive_html(filename)
        
        return filename

    @classmethod
    def save_comprehensive_html(cls, filename="mcts_comprehensive_visualization.html"):
        """
        Create a comprehensive HTML file with all trials and steps using JSON data.
        Includes trial selection, step selection, and navigation.
        """
        if not cls.global_trial_data:
            print("No global trial data to save.")
            return
            
        # Organize data by trial and step
        trials_data = {}
        json_snapshots = []
        
        for snapshot in cls.global_trial_data:
            trial_id = snapshot['trial_id']
            step_num = snapshot['step_number']
            
            if trial_id not in trials_data:
                trials_data[trial_id] = {}
            trials_data[trial_id][step_num] = snapshot
            
            # Prepare JSON snapshot data
            json_snapshots.append({
                "id": f"t{trial_id}_s{step_num}",
                "title": f"Trial {trial_id} - {snapshot['name']}",
                "trial_id": trial_id,
                "step_number": step_num,
                "total_nodes": snapshot['total_nodes'],
                "nodes": snapshot.get('json_nodes', []),
                "edges": snapshot.get('json_edges', []),
                "grid_size": snapshot.get('args', {}).get('n', 'unknown')
            })
        
        print(f"Organizing {len(cls.global_trial_data)} snapshots across {len(trials_data)} trials")
        
        # Also save JSON data to separate file
        import json
        json_filename = filename.replace('.html', '_data.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_snapshots, f, indent=2, ensure_ascii=False)
        print(f"JSON data saved to: {json_filename}")
        
        # Create the comprehensive HTML using the improved approach
        html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>MCTS Comprehensive Tree Visualization</title>
  <style>
    body {{ 
        margin: 0; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #1a1a1a;
        color: #ffffff;
    }}
    #toolbar {{ 
        display: flex; 
        gap: 12px; 
        align-items: center; 
        padding: 16px; 
        border-bottom: 1px solid #333; 
        background-color: #2d2d2d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    #mynetwork {{ 
        height: calc(100vh - 120px); 
        background-color: #222222;
        border-radius: 8px;
        margin: 16px;
        border: 1px solid #444;
    }}
    button, select {{ 
        padding: 8px 16px; 
        border-radius: 6px; 
        border: 1px solid #555; 
        background: #3a3a3a; 
        color: #ffffff;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 14px;
    }}
    button:hover, select:hover {{
        background: #4a4a4a;
    }}
    button:disabled {{
        background: #2a2a2a;
        color: #666;
        cursor: not-allowed;
    }}
    #title {{ 
        font-weight: 600; 
        margin-left: 16px; 
        font-size: 16px;
        color: #4CAF50;
    }}
    .stats {{
        display: flex;
        gap: 16px;
        margin-left: auto;
        font-size: 12px;
        color: #aaa;
    }}
    .stat-item {{
        background: #333;
        padding: 4px 8px;
        border-radius: 4px;
    }}
    .keyboard-help {{
        position: fixed;
        bottom: 16px;
        right: 16px;
        background: #333;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #aaa;
        border: 1px solid #444;
    }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="prev">← Prev Step</button>
    <button id="next">Next Step →</button>
    <select id="trialSelect">
        <option value="">Select Trial...</option>"""
        
        # Add trial options
        for trial_id in sorted(trials_data.keys()):
            trial_steps = len(trials_data[trial_id])
            html_content += f'<option value="{trial_id}">Trial {trial_id} ({trial_steps} steps)</option>'
        
        html_content += f"""
    </select>
    <select id="stepSelect">
        <option value="">Select Step...</option>
    </select>
    <button id="autoPlay">⏯ Auto Play</button>
    <span id="title">MCTS Tree Visualization</span>
    <div class="stats">
        <div class="stat-item">Nodes: <span id="nodeCount">-</span></div>
        <div class="stat-item">Grid: <span id="gridSize">-</span></div>
        <div class="stat-item">Step: <span id="currentStep">-</span></div>
    </div>
  </div>
  <div id="mynetwork"></div>
  <div class="keyboard-help">
    ⌨️ Use ← → for steps, ↑ ↓ for trials, Space for auto-play
  </div>

  <!-- vis-network (vis.js) -->
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script>
    const snapshots = {json.dumps(json_snapshots, ensure_ascii=False, indent=2)};
    let currentIdx = 0;
    let autoPlayInterval = null;
    let isAutoPlaying = false;

    // Initialize DataSet and Network
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const container = document.getElementById('mynetwork');
    const network = new vis.Network(container, {{ nodes, edges }}, {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 150,
          nodeSpacing: 100,
          treeSpacing: 200
        }}
      }},
      physics: {{
        hierarchicalRepulsion: {{
          centralGravity: 0.0,
          springLength: 100,
          springConstant: 0.01,
          nodeDistance: 120,
          damping: 0.09
        }},
        maxVelocity: 50,
        solver: "hierarchicalRepulsion",
        stabilization: {{iterations: 100}}
      }},
      nodes: {{
        font: {{ 
          size: 11, 
          color: '#ffffff',
          multi: false,
          align: 'center'
        }},
        borderWidth: 2,
        shadow: true,
        widthConstraint: {{ maximum: 150 }},
        heightConstraint: {{ minimum: 80 }}
      }},
      edges: {{
        color: {{ color: '#666666' }},
        smooth: {{
          type: "cubicBezier",
          forceDirection: "vertical",
          roundness: 0.4
        }},
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
      }},
      interaction: {{ hover: true }},
      configure: {{ enabled: false }}
    }});

    // Load a specific snapshot
    function loadSnapshot(idx) {{
      if (idx < 0 || idx >= snapshots.length) return;
      
      currentIdx = idx;
      const snapshot = snapshots[idx];
      
      // Clear and add new data
      nodes.clear();
      edges.clear();
      
      // Process nodes - convert label arrays to multi-line text
      if (snapshot.nodes && snapshot.nodes.length > 0) {{
        const processedNodes = snapshot.nodes.map(node => {{
          if (Array.isArray(node.label)) {{
            // Convert array of lines to newline-separated text for vis.js
            node.label = node.label.join('\\n');
          }}
          return node;
        }});
        nodes.add(processedNodes);
      }}
      if (snapshot.edges && snapshot.edges.length > 0) {{
        edges.add(snapshot.edges);
      }}
      
      // Update UI
      document.getElementById('title').textContent = snapshot.title;
      document.getElementById('nodeCount').textContent = snapshot.total_nodes;
      document.getElementById('gridSize').textContent = snapshot.grid_size;
      document.getElementById('currentStep').textContent = `${{snapshot.step_number + 1}}/${{getStepsForTrial(snapshot.trial_id)}}`;
      
      // Update dropdowns
      document.getElementById('trialSelect').value = snapshot.trial_id;
      updateStepDropdown(snapshot.trial_id);
      document.getElementById('stepSelect').value = snapshot.step_number;
      
      // Update button states
      updateButtons();
      
      // Fit the network view
      setTimeout(() => network.fit({{ animation: true }}), 100);
    }}
    
    function getStepsForTrial(trialId) {{
      return snapshots.filter(s => s.trial_id === trialId).length;
    }}
    
    function updateStepDropdown(trialId) {{
      const stepSelect = document.getElementById('stepSelect');
      stepSelect.innerHTML = '<option value="">Select Step...</option>';
      
      const trialSnapshots = snapshots.filter(s => s.trial_id === trialId).sort((a, b) => a.step_number - b.step_number);
      trialSnapshots.forEach(snapshot => {{
        const option = document.createElement('option');
        option.value = snapshot.step_number;
        option.textContent = `Step ${{snapshot.step_number + 1}} (${{snapshot.total_nodes}} nodes)`;
        stepSelect.appendChild(option);
      }});
    }}
    
    function updateButtons() {{
      const prevBtn = document.getElementById('prev');
      const nextBtn = document.getElementById('next');
      
      prevBtn.disabled = currentIdx <= 0;
      nextBtn.disabled = currentIdx >= snapshots.length - 1;
    }}
    
    function toggleAutoPlay() {{
      isAutoPlaying = !isAutoPlaying;
      const btn = document.getElementById('autoPlay');
      
      if (isAutoPlaying) {{
        btn.textContent = '⏸ Pause';
        autoPlayInterval = setInterval(() => {{
          if (currentIdx < snapshots.length - 1) {{
            loadSnapshot(currentIdx + 1);
          }} else {{
            toggleAutoPlay(); // Stop at end
          }}
        }}, 2000);
      }} else {{
        btn.textContent = '⏯ Auto Play';
        if (autoPlayInterval) {{
          clearInterval(autoPlayInterval);
          autoPlayInterval = null;
        }}
      }}
    }}

    // Event handlers
    document.getElementById('prev').onclick = () => loadSnapshot(currentIdx - 1);
    document.getElementById('next').onclick = () => loadSnapshot(currentIdx + 1);
    document.getElementById('autoPlay').onclick = toggleAutoPlay;
    
    document.getElementById('trialSelect').onchange = (e) => {{
      if (e.target.value) {{
        updateStepDropdown(e.target.value);
        const firstStep = snapshots.find(s => s.trial_id === e.target.value);
        if (firstStep) {{
          const idx = snapshots.indexOf(firstStep);
          loadSnapshot(idx);
        }}
      }}
    }};
    
    document.getElementById('stepSelect').onchange = (e) => {{
      const trialId = document.getElementById('trialSelect').value;
      if (trialId && e.target.value !== '') {{
        const stepNum = parseInt(e.target.value);
        const snapshot = snapshots.find(s => s.trial_id === trialId && s.step_number === stepNum);
        if (snapshot) {{
          const idx = snapshots.indexOf(snapshot);
          loadSnapshot(idx);
        }}
      }}
    }};

    // Keyboard controls
    document.addEventListener('keydown', (e) => {{
      switch(e.key) {{
        case 'ArrowLeft':
          e.preventDefault();
          loadSnapshot(currentIdx - 1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          loadSnapshot(currentIdx + 1);
          break;
        case 'ArrowUp':
          e.preventDefault();
          // Previous trial
          const currentTrial = snapshots[currentIdx]?.trial_id;
          const trials = [...new Set(snapshots.map(s => s.trial_id))].sort();
          const currentTrialIdx = trials.indexOf(currentTrial);
          if (currentTrialIdx > 0) {{
            const prevTrial = trials[currentTrialIdx - 1];
            const firstStepOfPrevTrial = snapshots.find(s => s.trial_id === prevTrial);
            if (firstStepOfPrevTrial) {{
              loadSnapshot(snapshots.indexOf(firstStepOfPrevTrial));
            }}
          }}
          break;
        case 'ArrowDown':
          e.preventDefault();
          // Next trial
          const currentTrial2 = snapshots[currentIdx]?.trial_id;
          const trials2 = [...new Set(snapshots.map(s => s.trial_id))].sort();
          const currentTrialIdx2 = trials2.indexOf(currentTrial2);
          if (currentTrialIdx2 < trials2.length - 1) {{
            const nextTrial = trials2[currentTrialIdx2 + 1];
            const firstStepOfNextTrial = snapshots.find(s => s.trial_id === nextTrial);
            if (firstStepOfNextTrial) {{
              loadSnapshot(snapshots.indexOf(firstStepOfNextTrial));
            }}
          }}
          break;
        case ' ':
          e.preventDefault();
          toggleAutoPlay();
          break;
      }}
    }});

    // Initialize
    if (snapshots.length > 0) {{
      loadSnapshot(0);
    }} else {{
      document.getElementById('title').textContent = 'No snapshots available';
    }}
  </script>
</body>
</html>
"""
        
        # Write the file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive visualization saved to: {filename}")
        print(f"Total snapshots: {len(cls.global_trial_data)}")
        print(f"Trials: {len(trials_data)}")
        
        return filename
        """
        Create an HTML file with all snapshots that allows switching between them.
        """
        if not self.snapshots:
            print("No snapshots to save.")
            return
        
        # Start building the HTML content
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MCTS Tree Snapshots</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #222;
            color: white;
        }
        .controls {
            margin-bottom: 20px;
            text-align: center;
            background-color: #333;
            padding: 15px;
            border-radius: 5px;
        }
        .controls button, .controls select {
            margin: 0 10px;
            padding: 10px 20px;
            font-size: 16px;
            background-color: #444;
            color: white;
            border: 1px solid #555;
            border-radius: 5px;
            cursor: pointer;
        }
        .controls button:hover, .controls select:hover {
            background-color: #555;
        }
        .snapshot-container {
            width: 100%;
            height: 600px;
            border: 1px solid #555;
            border-radius: 5px;
            display: none;
            background-color: #222;
        }
        .snapshot-container.active {
            display: block;
        }
        .info {
            text-align: center;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
        }
        .instructions {
            text-align: center;
            margin-bottom: 10px;
            font-style: italic;
            color: #aaa;
        }
    </style>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
    <h1 style="text-align: center;">MCTS Tree Visualization - Multi Snapshot View</h1>
    <div class="instructions">Use the controls below or arrow keys (← →) to navigate between snapshots</div>
    <div class="controls">
        <button onclick="previousSnapshot()">← Previous</button>
        <select id="snapshotSelect" onchange="selectSnapshot()">"""
        
        # Add snapshot options
        for i, snapshot in enumerate(self.snapshots):
            html_content += f'<option value="{i}">{snapshot["name"]}</option>'
        
        html_content += """
        </select>
        <button onclick="nextSnapshot()">Next →</button>
    </div>
    <div class="info">
        <span id="currentInfo">Snapshot 1 of """ + str(len(self.snapshots)) + """</span>
    </div>
    
"""
        
        # Add snapshot containers
        for i, snapshot in enumerate(self.snapshots):
            active_class = "active" if i == 0 else ""
            unique_id = f"snapshot{i}"
            html_content += f'''
    <div id="{unique_id}" class="snapshot-container {active_class}">
        <div id="vis{i}" style="width: 100%; height: 100%;"></div>
    </div>
'''
        
        # Add the JavaScript to create all networks
        html_content += '''
    <script>
        let currentSnapshot = 0;
        const totalSnapshots = ''' + str(len(self.snapshots)) + ''';
        let networks = [];
        
        // Network data for each snapshot
        const snapshotData = ['''
        
        # Add network data for each snapshot
        for i, snapshot in enumerate(self.snapshots):
            if i > 0:
                html_content += ','
            # Get the network from the snapshot and extract nodes and edges
            net = snapshot['network']
            nodes_data = []
            edges_data = []
            
            # Extract node and edge information
            # Note: This is a simplified approach. In a real implementation,
            # you might want to store the raw data separately.
            html_content += f'''
            {{
                "name": "{snapshot['name']}",
                "nodes": [],
                "edges": []
            }}'''
        
        html_content += '''
        ];
        
        function initializeNetworks() {
            // For now, create simple placeholder networks
            // In a full implementation, you would recreate the actual network data
            for (let i = 0; i < totalSnapshots; i++) {
                const container = document.getElementById('vis' + i);
                const nodes = new vis.DataSet([
                    {id: 'root', label: 'Loading...', x: 0, y: 0}
                ]);
                const edges = new vis.DataSet([]);
                const data = { nodes: nodes, edges: edges };
                const options = {
                    layout: {
                        hierarchical: {
                            enabled: true,
                            direction: "UD"
                        }
                    },
                    physics: {
                        enabled: false
                    }
                };
                networks[i] = new vis.Network(container, data, options);
            }
        }
        
        function showSnapshot(index) {
            // Hide all snapshots
            for (let i = 0; i < totalSnapshots; i++) {
                document.getElementById('snapshot' + i).classList.remove('active');
            }
            
            // Show selected snapshot
            document.getElementById('snapshot' + index).classList.add('active');
            document.getElementById('snapshotSelect').value = index;
            document.getElementById('currentInfo').textContent = 
                'Snapshot ' + (index + 1) + ' of ' + totalSnapshots + ' - ' + snapshotData[index].name;
                
            // Fit the network
            if (networks[index]) {
                setTimeout(() => networks[index].fit(), 100);
            }
        }
        
        function nextSnapshot() {
            currentSnapshot = (currentSnapshot + 1) % totalSnapshots;
            showSnapshot(currentSnapshot);
        }
        
        function previousSnapshot() {
            currentSnapshot = (currentSnapshot - 1 + totalSnapshots) % totalSnapshots;
            showSnapshot(currentSnapshot);
        }
        
        function selectSnapshot() {
            currentSnapshot = parseInt(document.getElementById('snapshotSelect').value);
            showSnapshot(currentSnapshot);
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowLeft') {
                previousSnapshot();
            } else if (event.key === 'ArrowRight') {
                nextSnapshot();
            }
        });
        
        // Initialize when page loads
        window.addEventListener('load', function() {
            initializeNetworks();
            showSnapshot(0);
        });
    </script>
</body>
</html>
'''
        
        # Save the file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Multi-snapshot visualization saved as: {filename}")
        print(f"Total snapshots: {len(self.snapshots)}")
        print("Note: This is a simplified version. Individual snapshots can be saved separately for full functionality.")

    def search(self, state):
        # define root
        if self.args.get('node_compression', False):
            root = Node_Compressed(self.game, self.args, state)
            print("Using Node_Compressed for MCTS", flush=True)
        else:
            root = Node(self.game, self.args, state)

        if self.args['process_bar'] == True:
            search_iterator = trange(self.args['num_searches'])
        else:
            search_iterator = range(self.args['num_searches'])

        # Pre-fetch Constant C
        const_C = self.args['C']
        num_searches = self.args['num_searches']

        for search in search_iterator:
            node = root

            # Optimization: Calculate decay once per search iteration
            ratio = search / num_searches
            exploration_factor = const_C * exploration_decay_nb(ratio)

            # selection
            while node.is_fully_expanded(): #         return self.is_full and len(self.children) > 0
                node = node.select(exploration_factor=exploration_factor)

            if node.action_taken is not None:
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)
                # has_collinear = self.game.check_collinear(node.state, node.action_taken)
                # value, _ = self.game.get_value_and_terminated(node.state)

                if not is_terminal:
                    node = node.expand()
                    value, terminal_state = node.simulate()
                    self.update_optimal_states(value, terminal_state)
            else:
                node = node.expand()
                value, terminal_state = node.simulate()
                self.update_optimal_states(value, terminal_state)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        
        # Add tree visualization if enabled
        if self.args.get('tree_visualization', False):
            # Create snapshot name based on current game state
            num_points = np.sum(state)
            snapshot_name = f"Step {num_points}: {num_points} points placed"
            
            # Display the tree
            self.tree_visualization(root, snapshot_name)
            
            # Prompt user for action probability output (only if pause_at_each_step is enabled)
            if self.args.get('pause_at_each_step', True):
                try:
                    response = input("Output action prob? (y/n): ").strip().lower()
                    if response == 'y':
                        print("Action probabilities:")
                        print(action_probs)
                except (EOFError, KeyboardInterrupt):
                    # Handle cases where input is not available (e.g., in automated runs)
                    pass
        
        return action_probs
            
            # expansion
            # simulation
            # backpropagation

        # return visit_counts

class MCTS_Tree_Reuse(MCTS):

    def __init__(self, game, args={'num_searches': 1000,'C': 1.4}):
        super().__init__(game, args)
        self.current_root = None

    def search(self, state):
        root = None
        # Try to find if we can reuse the current tree
        if self.current_root is not None:
             # Check children of current root to see if we moved to one of them
             matching_children = []
             for child in self.current_root.children:
                 if np.array_equal(child.state, state):
                     matching_children.append(child)
                     
             if matching_children:
                 # If duplicates exist (due to symmetry/transpositions), pick the most visited one
                 root = max(matching_children, key=lambda node: node.visit_count)
             
             # Also check if the state is the current root itself (e.g. searching same state again)
             if root is None and np.array_equal(self.current_root.state, state):
                 root = self.current_root

        if root is not None:
            # We found a reuseable subtree. 
            # Detach from parent to allow garbage collection of the old tree parts
            root.parent = None
            print(f"Tree Reuse: Reusing node with {root.visit_count} visits.", flush=True)
        else:
            # Create new root
            if self.args.get('node_compression', False):
                root = Node_Compressed(self.game, self.args, state)
                print("Using Node_Compressed for MCTS (New Tree)", flush=True)
            else:
                root = Node(self.game, self.args, state)
                print("Starting New Tree", flush=True)
        
        self.current_root = root

        # Determine how many more searches we need
        current_visits = root.visit_count
        target_searches = self.args['num_searches']
        remaining_searches = max(0, target_searches - current_visits)
        
        if remaining_searches < target_searches:
             print(f"Tree Reuse: Node has {current_visits}, running {remaining_searches} additional searches.", flush=True)
        
        if remaining_searches > 0:

            if self.args['process_bar'] == True:
                search_iterator = trange(remaining_searches, desc="MCTS Reuse", leave=False)
            else:
                search_iterator = range(remaining_searches)

            # Pre-fetch Constant C
            const_C = self.args['C']

            for search in search_iterator:
                node = root

                # Optimization with Tree Reuse Ratio
                ratio = (current_visits + search) / target_searches
                if ratio > 1.0: ratio = 1.0
                exploration_factor = const_C * exploration_decay_nb(ratio)

                # selection
                while node.is_fully_expanded():
                    node = node.select(exploration_factor=exploration_factor)

                if node.action_taken is not None:
                    value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)

                    if not is_terminal:
                        node = node.expand()
                        value, terminal_state = node.simulate()
                        self.update_optimal_states(value, terminal_state)
                else:
                    node = node.expand()
                    value, terminal_state = node.simulate()
                    self.update_optimal_states(value, terminal_state)

                node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        
        # Add tree visualization if enabled
        if self.args.get('tree_visualization', False):
            # Create snapshot name based on current game state
            num_points = np.sum(state)
            snapshot_name = f"Step {num_points}: {num_points} points placed"
            
            # Display the tree
            self.tree_visualization(root, snapshot_name)
            
            # Prompt user for action probability output (only if pause_at_each_step is enabled)
            if self.args.get('pause_at_each_step', True):
                try:
                    response = input("Output action prob? (y/n): ").strip().lower()
                    if response == 'y':
                        print(f"Action probs: {action_probs}")
                except (EOFError, KeyboardInterrupt):
                    pass
        
        return action_probs

