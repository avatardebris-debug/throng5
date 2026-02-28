"""
mcts_planner.py — Numpy MCTS over WorldModel
=============================================

Pure-numpy UCT search wired to throng4's WorldModel dynamics.

Adapted from muzero-main/muzero/mcts.py (Apache 2.0, Michael Hu 2022).
All PyTorch dependencies removed; MuZeroNet replaced with WorldModel.predict().

Architecture:
  - Node / MinMaxStats: identical to MuZero source (no changes needed)
  - uct_search: replaces network.initial_inference / recurrent_inference
                with WorldModel.predict(state, action) -> (next_state, reward)
  - Prior probs: uniform (1/n_actions) by default; accepts Q-values from
                 PortableNNAgent to bias search toward learned actions

Usage:
    from throng4.basal_ganglia.mcts_planner import MCTSPlanner
    from throng4.basal_ganglia.dreamer_engine import WorldModel

    wm      = WorldModel(state_size=64, n_actions=18)
    planner = MCTSPlanner(world_model=wm, n_actions=18, n_simulations=50)

    # Returns (best_action, visit_probs, root_Q)
    action, probs, value = planner.search(current_state_compressed)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ── Min-max normaliser (verbatim from MuZero) ─────────────────────────────────

_INF = float("inf")


class MinMaxStats:
    """Tracks min/max Q-values seen during a single MCTS run for normalisation."""

    def __init__(self) -> None:
        self.maximum = -_INF
        self.minimum =  _INF

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalise(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


# ── Search tree node (adapted from MuZero, board-game logic stripped) ─────────

class Node:
    """Node in the UCT search tree."""

    __slots__ = ("prior", "move", "parent", "is_expanded",
                 "N", "W", "reward", "hidden_state", "children")

    def __init__(self, prior: float = 0.0, move: int = -1,
                 parent: "Node | None" = None) -> None:
        self.prior        = prior
        self.move         = move          # action that led to this node
        self.parent       = parent
        self.is_expanded  = False
        self.N            = 0             # visit count
        self.W            = 0.0           # total value
        self.reward       = 0.0           # immediate reward at this node
        self.hidden_state: Optional[np.ndarray] = None
        self.children:     List[Node]    = []

    def expand(self, prior: np.ndarray, hidden_state: np.ndarray,
               reward: float) -> None:
        """Expand all actions from this node."""
        if self.is_expanded:
            raise RuntimeError("Node already expanded.")
        self.hidden_state = hidden_state
        self.reward       = reward
        for a in range(prior.shape[0]):
            self.children.append(Node(prior=float(prior[a]), move=a, parent=self))
        self.is_expanded = True

    # ── UCB selection ─────────────────────────────────────────────────────────

    def best_child(self, stats: MinMaxStats, pb_c_base: float,
                   pb_c_init: float, discount: float) -> "Node":
        """Return child with highest Q + U score."""
        if not self.is_expanded:
            raise ValueError("Expand leaf node first.")
        q_vals = self._child_Q(stats, discount)
        u_vals = self._child_U(pb_c_base, pb_c_init)
        idx    = np.argmax(q_vals + u_vals)
        return self.children[idx]

    def _child_Q(self, stats: MinMaxStats, discount: float) -> np.ndarray:
        return np.array([
            stats.normalise(c.reward + discount * c.Q) if c.N > 0 else 0.0
            for c in self.children
        ], dtype=np.float32)

    def _child_U(self, pb_c_base: float, pb_c_init: float) -> np.ndarray:
        exploration = math.log((self.N + pb_c_base + 1) / pb_c_base) + pb_c_init
        sqrt_N      = math.sqrt(self.N)
        return np.array([
            c.prior * exploration * sqrt_N / (c.N + 1)
            for c in self.children
        ], dtype=np.float32)

    # ── Backup ────────────────────────────────────────────────────────────────

    def backup(self, value: float, stats: MinMaxStats, discount: float) -> None:
        """Propagate value up to root, updating N, W, and MinMaxStats."""
        current = self
        while current is not None:
            current.W += value
            current.N += 1
            stats.update(current.reward + discount * current.Q)
            value   = current.reward + discount * value
            current = current.parent

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def child_N(self) -> np.ndarray:
        return np.array([c.N for c in self.children], dtype=np.int32)


# ── MCTSPlanner ───────────────────────────────────────────────────────────────

class MCTSPlanner:
    """
    UCT tree search over throng4's WorldModel.

    Args:
        world_model:    A WorldModel instance (must have .predict(state, action)).
        n_actions:      Number of discrete actions.
        n_simulations:  MCTS simulations per search call (default 50).
        discount:       Future reward discount γ (default 0.97).
        pb_c_base:      UCB exploration base constant (MuZero default 19652).
        pb_c_init:      UCB exploration init constant (MuZero default 1.25).
        dirichlet_eps:  Exploration noise weight at root (0 = off).
        dirichlet_alpha: Dirichlet noise concentration (0.3 = Atari default).
        value_depth:    Steps to roll out for leaf value estimate (0 = use 0).
        prior_fn:       Optional callable (state) -> np.ndarray of shape (n_actions,)
                        returning prior probabilities. If None, uses uniform.
        lethal_state_fn: Optional callable (state) -> bool returning True if the
                         state is guaranteed fatal. MCTS will hard-prune it (-10000).
    """

    def __init__(
        self,
        world_model,
        n_actions:       int   = 18,
        n_simulations:   int   = 50,
        discount:        float = 0.97,
        pb_c_base:       float = 19652.0,
        pb_c_init:       float = 1.25,
        dirichlet_eps:   float = 0.25,
        dirichlet_alpha: float = 0.3,
        value_depth:     int   = 5,
        prior_fn         = None,
        lethal_state_fn  = None,
    ) -> None:
        self.wm             = world_model
        self.n_actions      = n_actions
        self.n_sims         = n_simulations
        self.discount       = discount
        self.pb_c_base      = pb_c_base
        self.pb_c_init      = pb_c_init
        self.dir_eps        = dirichlet_eps
        self.dir_alpha      = dirichlet_alpha
        self.value_depth    = value_depth
        self.prior_fn       = prior_fn
        self.lethal_fn      = lethal_state_fn

        # Stats
        self._search_count  = 0
        self._total_ms      = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        temperature:   float = 1.0,
        action_mask:   "np.ndarray | None" = None,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Run UCT search from the given (compressed) state.

        Args:
            state:       Compressed state vector.
            deterministic: If True, pick the most visited action.
            temperature:   Action sampling temperature.
            action_mask:   Boolean array of shape (n_actions,).
                           False = action is forbidden (banned at root level).
                           Applied BEFORE simulations so MCTS never wastes
                           simulations on banned branches.
        """
        import time
        t0 = time.perf_counter()

        stats    = MinMaxStats()
        root     = Node(prior=0.0)
        prior    = self._get_prior(state)

        # Apply action mask (real-coordinate danger zones, pre-simulation)
        if action_mask is not None:
            prior = prior.copy()
            prior[~action_mask] = 0.0
            total = prior.sum()
            if total > 0:
                prior /= total
            else:
                prior = np.ones(self.n_actions, dtype=np.float32) / self.n_actions

        # Apply action preference boost (real-coordinate zone bias).
        # self.action_boost is set externally each step from room_constants.
        # Boost is additive then renormalized — strongly favours jump-right on
        # the conveyor belt even before Q-values have learned that action.
        _boost = getattr(self, "action_boost", None)
        if _boost is not None and _boost.any():
            prior = prior.copy()
            prior = prior + _boost
            # Re-zero any still-masked actions
            if action_mask is not None:
                prior[~action_mask] = 0.0
            total = prior.sum()
            if total > 0:
                prior /= total

        # Add Dirichlet noise at root for exploration
        if self.dir_eps > 0:
            noise = np.random.dirichlet(np.ones(self.n_actions) * self.dir_alpha)
            prior = (1 - self.dir_eps) * prior + self.dir_eps * noise

        # Zero out masked actions from noise too
        if action_mask is not None:
            prior[~action_mask] = 0.0
            total = prior.sum()
            if total > 0:
                prior /= total

        root.expand(prior, hidden_state=state.copy(), reward=0.0)

        for _ in range(self.n_sims):
            # ── Phase 1: Select leaf ──────────────────────────────────────────
            node = root
            while node.is_expanded:
                node = node.best_child(stats, self.pb_c_base,
                                       self.pb_c_init, self.discount)

            # ── Phase 2: Expand leaf via WorldModel ───────────────────────────
            parent_state = node.parent.hidden_state  # type: ignore[union-attr]
            next_state, reward = self.wm.predict(parent_state, node.move)

            # Hard-prune lethal states predicted by the world model
            is_lethal = False
            if self.lethal_fn is not None and self.lethal_fn(next_state):
                reward = -10000.0
                is_lethal = True

            leaf_prior = self._get_prior(next_state)
            node.expand(leaf_prior, hidden_state=next_state, reward=reward)

            # ── Phase 3: Leaf value estimate (short rollout) ──────────────────
            value = -10000.0 if is_lethal else self._rollout_value(next_state)

            # ── Phase 4: Backup ───────────────────────────────────────────────
            node.backup(value, stats, self.discount)

        # ── Generate action from visit counts ─────────────────────────────────
        visits = root.child_N                       # shape (n_actions,)
        
        # Mask out known lethal branches (Q < -9000) so they get 0 visits
        for i, child in enumerate(root.children):
            if child.Q < -9000.0:
                visits[i] = 0

        valid_sum = visits.sum()
        if valid_sum == 0:
            # Fallback if ALL branches are lethal (unlikely, but failsafe)
            visits = np.ones(self.n_actions)
            valid_sum = self.n_actions

        if deterministic or temperature == 0.0:
            action = int(np.argmax(visits))
        else:
            exp     = max(1.0, min(5.0, 1.0 / max(temperature, 1e-6)))
            weights = np.power(visits.astype(np.float64), exp)
            probs   = weights / weights.sum()
            action  = int(np.random.choice(self.n_actions, p=probs))

        visit_probs = visits / valid_sum

        elapsed = (time.perf_counter() - t0) * 1000
        self._search_count += 1
        self._total_ms     += elapsed

        return action, visit_probs.astype(np.float32), root.Q

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_prior(self, state: np.ndarray) -> np.ndarray:
        """Return prior probabilities over actions for this state."""
        if self.prior_fn is not None:
            try:
                raw = np.asarray(self.prior_fn(state), dtype=np.float32)
                raw = raw - raw.max()           # numerical stability
                exp = np.exp(raw)
                return (exp / exp.sum()).astype(np.float32)
            except Exception:
                pass
        return np.ones(self.n_actions, dtype=np.float32) / self.n_actions

    def _rollout_value(self, state: np.ndarray) -> float:
        """Estimate leaf value with a short random rollout."""
        if self.value_depth <= 0:
            return 0.0
        total  = 0.0
        gamma  = 1.0
        s      = state.copy()
        for _ in range(self.value_depth):
            a          = np.random.randint(self.n_actions)
            s, r       = self.wm.predict(s, a)
            total     += gamma * r
            gamma     *= self.discount
        return total

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def avg_search_ms(self) -> float:
        return self._total_ms / max(1, self._search_count)

    def summary(self) -> str:
        return (f"MCTSPlanner: {self.n_sims} sims | "
                f"{self.n_actions} actions | "
                f"avg {self.avg_search_ms:.1f}ms/search | "
                f"{self._search_count} searches run")
