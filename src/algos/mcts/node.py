import numpy as np
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor
from src.algos.mcts.utils import (
    exploration_decay_nb,
    _pack_bits_bool2d,
    _unpack_bits_to_2d,
    _bit_clear_inplace,
    _bit_set_inplace,
    get_valid_moves_subset_nb
)
from src.algos.mcts.simulation import simulate_nb, simulate_with_priority_nb


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)

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
                    parent.state, parent.action_space, self.action_taken)
                self.valid_moves = game.filter_valid_moves_by_symmetry(
                    self.action_space, state
                ).copy()
            else:
                self.valid_moves = game.get_valid_moves_subset(
                    parent.state, parent.action_space, self.action_taken)
                self.action_space = self.valid_moves.copy()

        # Ensure action_space is immutable
        self.action_space.flags.writeable = False

        self.is_full = False
        self._cached_ucb = None     # Cached UCB value
        self._ucb_dirty = True      # Indicates whether the cached UCB is stale

    def apply_virtual_loss(self):
        with self.lock:
            self.value_sum -= self._vl
            self.visit_count += 1
            self._ucb_dirty = True  # Mark UCB as outdated

    def revert_virtual_loss(self):
        with self.lock:
            self.value_sum += self._vl
            self._ucb_dirty = True  # Mark UCB as outdated

    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0

    def select(self, iter):
        best_child = None
        best_ucb = -np.inf
        log_N = math.log(self.visit_count)

        for child in self.children:
            ucb = self.get_ucb(child, iter, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, iter, log_N=None):
        if log_N is None:
            log_N = math.log(self.visit_count)

        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb

            q_value = child.value_sum / child.visit_count
            T_i = self.args['C'] * exploration_decay_nb(iter/self.args['num_searches'])
            exploration_value = T_i * math.sqrt(log_N / child.visit_count)
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
        child_state = self.game.get_next_state(child_state, action)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        tmp = self.state.copy()
        if self.args["simulate_with_priority"] == True:
            return simulate_with_priority_nb(tmp,
                                            self.game.row_count,
                                            self.game.column_count,
                                            self.game.pts_upper_bound,
                                            self.game.priority_grid,
                                            self.args['TopN'])
        else:
            return simulate_nb(tmp,
                            self.game.row_count,
                            self.game.column_count,
                            self.game.pts_upper_bound)

    def backpropagate(self, value):
        with self.lock:
            self.value_sum += value
            self._ucb_dirty = True  # Mark UCB as outdated
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)


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
        '_rows','_cols','_state_bits','_valid_bits','_action_bits'
    )

    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)

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
                action_space = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken)
                valid_moves  = game.filter_valid_moves_by_symmetry(action_space, self.state,).copy()  # self.state property unpacks
            else:
                valid_moves  = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken)
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
            self.value_sum -= self._vl
            self.visit_count += 1
            self._ucb_dirty = True

    def revert_virtual_loss(self):
        with self.lock:
            self.value_sum += self._vl
            self._ucb_dirty = True

    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0

    def select(self, iter):
        best_child = None
        best_ucb = -np.inf
        # avoid log(0)
        log_N = math.log(self.visit_count) if self.visit_count > 0 else 0.0

        for child in self.children:
            ucb = self.get_ucb(child, iter, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, iter, log_N=None):
        if log_N is None:
            log_N = math.log(self.visit_count) if self.visit_count > 0 else 0.0

        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb

            q_value = child.value_sum / max(1, child.visit_count)
            T_i = self.args['C'] * exploration_decay_nb(iter/self.args['num_searches'])
            exploration_value = T_i * math.sqrt(max(1e-12, log_N) / max(1, child.visit_count))
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
        child_state = self.game.get_next_state(child_state, action)

        # Create child node (compressed)
        child = Node_Compressed(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        # Unpack to 2D array; simulation mutates a copy
        tmp = self.state.copy()
        if self.args.get("simulate_with_priority", False):
            return simulate_with_priority_nb(tmp,
                                            self.game.row_count,
                                            self.game.column_count,
                                            self.game.pts_upper_bound,
                                            self.game.priority_grid,
                                            self.args['TopN'])
        else:
            return simulate_nb(tmp,
                            self.game.row_count,
                            self.game.column_count,
                            self.game.pts_upper_bound)

    def backpropagate(self, value):
        with self.lock:
            self.value_sum += value
            self._ucb_dirty = True
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)


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
