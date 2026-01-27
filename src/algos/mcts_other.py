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
from src.envs import N3il, N3il_with_symmetry, N3il_with_symmetry_and_symmetric_actions, supnorm_priority, supnorm_priority_array
from src.algos import MCTS, Node, Node_Compressed
from src.utils.symmetry import get_d4_orbit

import io
import base64
from pyvis.network import Network

import psutil
import os
import warnings

from src.rewards.n3il_rewards import get_value_nb

class ParallelMCTS(MCTS):
    def __init__(self, game, args):
        super().__init__(game, args)
        self.num_workers   = args.get('num_workers', 4)
        self.virtual_loss  = args.get('virtual_loss', 1.0)
        self.args = args

    # --- single simulation --------------------------------------------------
    def _search_once(self, root, worker_iter):
        path = []
        node = root

        # 1. SELECTION
        while node.is_fully_expanded():
            path.append(node)
            node.apply_virtual_loss()         # <‑‑ reserve
            node = node.select(iter=worker_iter*self.num_workers)

        # 2. EXPANSION / SIMULATION
        if node.action_taken is None:
            node = node.expand()
        path.append(node)
        node.apply_virtual_loss()             # reserve leaf
        
        # Simulate
        is_term = self.game.get_value_and_terminated(node.state, node.valid_moves)[1]
        if not is_term:
            value, _ = node.simulate()
        else:
            value = self.game.get_value_and_terminated(node.state, node.valid_moves)[0]

        # 3. UNDO VIRTUAL LOSS + BACKPROP
        for n in path:
            n.revert_virtual_loss()
        node.backpropagate(value)
    # ------------------------------------------------------------------------

    # --- parallel driver ----------------------------------------------------
    def search(self, state):
        root = Node(self.game, self.args, state)

        sims_per_worker = self.args['num_searches'] // self.num_workers
        remainder       = self.args['num_searches'] %  self.num_workers

        def worker(n_sims, worker_id=0):
            # Set deterministic seed for this worker thread
            if 'random_seed' in self.args:
                worker_seed = self.args['random_seed'] + worker_id * 10000
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            
            if self.args['process_bar'] == True:
                for worker_iter in trange(n_sims):
                    self._search_once(root, worker_iter)
            else:
                for worker_iter in range(n_sims):
                    self._search_once(root, worker_iter)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(worker, sims_per_worker, worker_id)
                       for worker_id in range(self.num_workers)]
            if remainder:                     # handle leftovers
                futures.append(pool.submit(worker, remainder, self.num_workers))
            wait(futures)

        # convert visit counts → prob. vector
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    

# ---------- Leaf/Child Paralllel MCTS ------------------------------------------------

def _rollout_many(child, R: int):
    """Run child.simulate() R times in the same worker thread; return the list of values."""
    if R <= 1:
        val, _ = child.simulate()
        return [val]
    vals = []
    for _ in range(R):
        val, _ = child.simulate()
        vals.append(val)
    return vals

# --- Child-parallel expansion node ---
class LeafChildParallelNode(Node):
    """
    Node class that expands all children in parallel (child-parallel expansion).
    """
    def __init__(self, game, args, state, parent=None, action_taken=None):
        super().__init__(game, args, state, parent, action_taken)

    def expand(self):
        """
        Expand all valid children in parallel and return a randomly chosen child.
        Important: do not mutate self.valid_moves until children are constructed,
        because Node.__init__ of children depends on parent's valid mask.
        """
        # Snapshot the parent's valid mask first
        with self.lock:
            valid_indices = np.where(self.valid_moves == 1)[0]
            if len(valid_indices) == 0:
                self.is_full = True
                return None
            parent_valid_snapshot = self.valid_moves.copy()
            parent_state = self.state  # read-only usage below

        # Build children in parallel
        def build_child(action):
            # Construct child state
            child_state = parent_state.copy()
            child_state = self.game.get_next_state(child_state, action)
            # Create child node (this will compute valid moves once)
            child = LeafChildParallelNode(self.game, self.args, child_state, self, action)
            # Optionally, we can override with our own subset computation using the snapshot:
            # This avoids recomputing if Node.__init__ is heavy or if we want to be explicit.
            try:
                child_valid = get_valid_moves_subset_nb(
                    parent_state,
                    parent_valid_snapshot,
                    action,
                    self.game.row_count,
                    self.game.column_count
                )
                child.valid_moves = child_valid
            except Exception:
                # Fallback: keep whatever Node.__init__ computed
                pass
            return child

        children = []
        with ThreadPoolExecutor(max_workers=min(len(valid_indices), 32)) as executor:
            futures = [executor.submit(build_child, a) for a in valid_indices]
            for fut in futures:
                children.append(fut.result())

        # Append children and then mark fully expanded
        with self.lock:
            self.children.extend(children)
            # Now it is safe to mark all these actions as used
            self.valid_moves[valid_indices] = 0
            self.is_full = True

        # Return a random child for compatibility with the base MCTS flow
        return random.choice(children) if children else None

class LeafChildParallelMCTS(MCTS):
    """
    MCTS variant that supports both leaf-parallel and child-parallel simulation strategies.
    - Leaf-parallel: Run multiple rollouts from a leaf before backpropagation.
    - Child-parallel: When expanding a node, run one simulation from each child in parallel.
      (If both are set, runs multiple rollouts per child in parallel.)
    """
    def __init__(self, game, args):
        super().__init__(game, args)
        self.num_workers = args.get('num_workers', 4)
        self.simulations_per_leaf = args.get('simulations_per_leaf', 1)
        self.child_parallel = args.get('child_parallel', True)
        self.virtual_loss = args.get('virtual_loss', 1.0)
        self.args = args

    def _simulate_leaf_parallel(self, node, num_simulations):
        """
        Run multiple rollouts from a single leaf; return list of values.
        No backprop here; caller will backprop in main thread.
        """
        if num_simulations <= 1:
            val, _ = node.simulate()
            return [val]
        # Usually we don't need virtual loss here, selection is single-threaded
        # We need a wrapper to unpack tuple for list comp comprehension if using map/comp
        # Or just use a loop
        
        def run_sim():
            val, _ = node.simulate()
            return val

        with ThreadPoolExecutor(max_workers=min(num_simulations, self.num_workers)) as pool:
            futures = [pool.submit(run_sim) for _ in range(num_simulations)]
            return [f.result() for f in futures]

    def _simulate_child_parallel(self, node, num_simulations):
        """
        Expand all valid children and run multiple rollouts per child in parallel.
        Returns a list of tuples: (child, [values...]).
        """
        # Snapshot parent's valid mask to avoid mutation races
        with node.lock:
            valid_indices = np.where(node.valid_moves == 1)[0]
            if valid_indices.size == 0:
                node.is_full = True
                return []
            parent_valid_snapshot = node.valid_moves.copy()
            parent_state = node.state

        # Build children in parallel first (no backprop here)
        def build_child(action):
            child_state = parent_state.copy()
            child_state = self.game.get_next_state(child_state, action)
            child = LeafChildParallelNode(self.game, self.args, child_state, node, action)
            # Override child's valid mask using snapshot so we don't recompute later
            try:
                child_valid = get_valid_moves_subset_nb(
                    parent_state,
                    parent_valid_snapshot,
                    action,
                    self.game.row_count,
                    self.game.column_count
                )
                child.valid_moves = child_valid
            except Exception:
                pass
            return child

        children = []
        with ThreadPoolExecutor(max_workers=min(len(valid_indices), self.num_workers)) as pool:
            futures = [pool.submit(build_child, a) for a in valid_indices]
            for fut in futures:
                children.append(fut.result())

        # Attach children and mark node fully expanded
        with node.lock:
            node.children.extend(children)
            node.valid_moves[valid_indices] = 0
            node.is_full = True

        # Now run R rollouts per child in parallel; collect results (no backprop here)
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futs = [pool.submit(_rollout_many, child, num_simulations) for child in children]
            for child, fut in zip(children, futs):
                vals = fut.result()
                results.append((child, vals))

        # Backprop only in the main thread to avoid data races
        for child, vals in results:
            for v in vals:
                child.backpropagate(v)

        return results

    def search(self, state):
        root = LeafChildParallelNode(self.game, self.args, state)
        num_searches = self.args.get('num_searches', 1000)
        process_bar = self.args.get('process_bar', False)
        search_iterator = trange(num_searches) if process_bar else range(num_searches)

        for search in search_iterator:
            node = root
            # 1) Selection (single-threaded)
            while node.is_fully_expanded():
                node = node.select(iter=search)

            # 2) If child-parallel enabled and node not yet expanded → expand + run
            if self.child_parallel and not node.is_fully_expanded():
                self._simulate_child_parallel(node, self.simulations_per_leaf)
                # Everything done inside; continue to next simulation
                continue

            # 3) Otherwise do leaf-parallel on a single child
            # If node is not root and terminal, just backprop its terminal value
            if node.action_taken is not None:
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)
                if is_terminal:
                    node.backpropagate(value)
                    continue
                # Expand one child (this expand returns a random child)
                node = node.expand()
                if node is None:
                    # No child could be expanded; treat as terminal with zero
                    continue
                values = self._simulate_leaf_parallel(node, self.simulations_per_leaf)
            else:
                # Root case: expand first
                node = node.expand()
                if node is None:
                    continue
                values = self._simulate_leaf_parallel(node, self.simulations_per_leaf)

            # Backprop the leaf-parallel values
            for v in values:
                node.backpropagate(v)

        # 4) Action probabilities from root's children
        action_probs = np.zeros(self.game.action_size)
        total = 0
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
            total += child.visit_count
        if total > 0:
            action_probs /= total
        return action_probs


class MCGSNode:
    """Node class for Monte Carlo Graph Search"""
    def __init__(self, state, state_key):
        self.state = state
        self.state_key = state_key
        self.lower_bound = 0.0
        self.upper_bound = 1.0  # V_max equivalent
        self.outgoing_edges = {}  # action -> (reward, next_state_key)
        self.has_outgoing_edges = False

class MCGS:
    """Monte Carlo Graph Search implementation compatible with MCTS interface"""
    def __init__(self, game, args={
        'num_searches': 1000,
        'C': 1.4,
        'gamma': 0.99  # discount factor for MCGS
    }):
        self.game = game
        self.args = args
        self.gamma = args.get('gamma', 0.99)
        self.V_max = 1.0 / (1.0 - self.gamma)
        self.graph = {}  # state_key -> MCGSNode
        
    def search(self, state):
        """Main MCGS algorithm following the pseudocode"""
        # Initialize graph with root node
        root_key = self.game.state_to_key(state)
        if root_key not in self.graph:
            self.graph[root_key] = MCGSNode(state.copy(), root_key)
        
        budget = self.args.get('num_searches', 1000)
        
        if self.args.get('process_bar', False):
            search_iterator = trange(budget)
        else:
            search_iterator = range(budget)
            
        for n in search_iterator:
            # Step 1: Bellman backups to compute value bounds
            self._compute_value_bounds()
            
            # Step 2: Optimistic sampling - follow path with highest upper bounds
            leaf_key = self._optimistic_sampling(root_key)
            
            # Step 3: Node expansion - add all possible actions from leaf
            if leaf_key in self.graph:
                self._expand_node(leaf_key)
        
        # Step 4: Return action probabilities based on lower bounds
        return self._get_action_probabilities(root_key)
    
    def _compute_value_bounds(self):
        """Compute lower and upper value bounds using Bellman operators"""
        # Initialize bounds
        for node in self.graph.values():
            if not node.has_outgoing_edges:
                # Sink nodes: evaluate using the game's evaluation function
                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, self.game.get_valid_moves(node.state)
                )
                if is_terminal:
                    node.lower_bound = value
                    node.upper_bound = value
                else:
                    node.lower_bound = 0.0
                    node.upper_bound = self.V_max
        
        # Iterate Bellman operators until convergence
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            old_lower = {k: v.lower_bound for k, v in self.graph.items()}
            old_upper = {k: v.upper_bound for k, v in self.graph.items()}
            
            # Update internal nodes
            for node in self.graph.values():
                if node.has_outgoing_edges:
                    # Lower bound update
                    max_lower = -np.inf
                    max_upper = -np.inf
                    
                    for action, (reward, next_key) in node.outgoing_edges.items():
                        if next_key in self.graph:
                            next_node = self.graph[next_key]
                            q_lower = reward + self.gamma * next_node.lower_bound
                            q_upper = reward + self.gamma * next_node.upper_bound
                            max_lower = max(max_lower, q_lower)
                            max_upper = max(max_upper, q_upper)
                    
                    node.lower_bound = max_lower if max_lower > -np.inf else 0.0
                    node.upper_bound = max_upper if max_upper > -np.inf else self.V_max
            
            # Check convergence
            converged = True
            for key in self.graph:
                if (abs(self.graph[key].lower_bound - old_lower[key]) > tolerance or
                    abs(self.graph[key].upper_bound - old_upper[key]) > tolerance):
                    converged = False
                    break
            
            if converged:
                break
    
    def _optimistic_sampling(self, start_key):
        """Follow optimistic policy to reach a leaf node"""
        current_key = start_key
        
        while current_key in self.graph and self.graph[current_key].has_outgoing_edges:
            node = self.graph[current_key]
            
            # Select action with highest upper bound
            best_action = None
            best_value = -np.inf
            
            for action, (reward, next_key) in node.outgoing_edges.items():
                if next_key in self.graph:
                    next_node = self.graph[next_key]
                    q_value = reward + self.gamma * next_node.upper_bound
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
            
            if best_action is not None:
                _, current_key = node.outgoing_edges[best_action]
            else:
                break
        
        return current_key
    
    def _expand_node(self, node_key):
        """Expand a leaf node by trying all possible actions"""
        if node_key not in self.graph:
            return
            
        node = self.graph[node_key]
        valid_moves = self.game.get_valid_moves(node.state)
        valid_actions = np.where(valid_moves == 1)[0]
        
        if len(valid_actions) == 0:
            return
        
        # Try all valid actions
        for action in valid_actions:
            # Generate next state using the game model
            next_state = node.state.copy()
            next_state = self.game.get_next_state(next_state, action)
            next_key = self.game.state_to_key(next_state)
            
            # Compute immediate reward (difference in game value)
            old_value, _ = self.game.get_value_and_terminated(
                node.state, self.game.get_valid_moves(node.state)
            )
            new_value, _ = self.game.get_value_and_terminated(
                next_state, self.game.get_valid_moves(next_state)
            )
            reward = new_value - old_value
            
            # Add edge to graph
            node.outgoing_edges[action] = (reward, next_key)
            
            # Add next state to graph if not already present
            if next_key not in self.graph:
                self.graph[next_key] = MCGSNode(next_state, next_key)
        
        node.has_outgoing_edges = True
    
    def _get_action_probabilities(self, root_key):
        """Convert lower bound Q-values to action probabilities"""
        action_probs = np.zeros(self.game.action_size)
        
        if root_key not in self.graph:
            return action_probs
        
        root_node = self.graph[root_key]
        
        if not root_node.has_outgoing_edges:
            # If no outgoing edges, return uniform over valid moves
            valid_moves = self.game.get_valid_moves(root_node.state)
            valid_actions = np.where(valid_moves == 1)[0]
            if len(valid_actions) > 0:
                for action in valid_actions:
                    action_probs[action] = 1.0 / len(valid_actions)
            return action_probs
        
        # Compute Q-values using lower bounds (conservative estimate)
        q_values = {}
        for action, (reward, next_key) in root_node.outgoing_edges.items():
            if next_key in self.graph:
                next_node = self.graph[next_key]
                q_values[action] = reward + self.gamma * next_node.lower_bound
            else:
                q_values[action] = reward
        
        if not q_values:
            return action_probs
        
        # Convert to probabilities (softmax-like but focused on best actions)
        max_q = max(q_values.values())
        actions_with_max_q = [a for a, q in q_values.items() if abs(q - max_q) < 1e-6]
        
        # Give equal probability to all actions with maximum Q-value
        prob_per_action = 1.0 / len(actions_with_max_q)
        for action in actions_with_max_q:
            action_probs[action] = prob_per_action
        
        return action_probs
