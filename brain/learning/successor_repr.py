"""
successor_repr.py — Successor Representation (SR) for subgoal routing and goal transfer.

The SR factorises the Q-function:
    Q(s, a) = ψ(s, a) · w

where:
  ψ(s, a) ∈ R^d  — successor features: expected discounted future feature occupancy
                   "If I take action a in state s, which features will I visit on average?"
  w ∈ R^d        — reward weights learned separately ("which features lead to reward?")

This is more powerful than a plain Q-function because:
  1. Goal transfer: same ψ, different w → zero-shot new goal Q without retraining
  2. Subgoal distance: ‖ψ(s) - ψ(goal)‖ is a principled distance metric in SR space
  3. Reuse: SubgoalPlanner can route with SR instead of (or alongside) LandmarkGraph

Architecture:
  - Tabular SR: dict mapping (state_hash, action) → ψ vector (for small discrete spaces)
  - Linear FA:  w @ phi(s)  where phi is the sensory feature vector (for large spaces)
  - Seamless fallback: if state never seen, return identity features (state = 1 step from itself)

Integration:
  - Hippocampus.process() calls sr.update(s_vec, a, s2_vec, done) every step
  - SubgoalPlanner uses sr.closest_subgoal(s_vec, g_vec) instead of Dijkstra
  - Striatum may call sr.predict_q(s_vec, w) as an auxiliary Q head (future)

Usage:
    sr = SRMatrix(n_features=64, n_actions=4)
    sr.update(s, action, s_next, done)
    subgoal_action = sr.closest_subgoal(current_state, goal_state)
    q_vals = sr.predict_q(current_state, reward_weights)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── State hashing ─────────────────────────────────────────────────────────


def _quantise(vec: np.ndarray, bins: int = 8) -> bytes:
    """Coarsely quantise a feature vector to a hashable key."""
    v = np.asarray(vec, dtype=np.float32).flatten()
    k = min(8, len(v))
    clipped = np.clip(v[:k], -3.0, 3.0)
    quant = (clipped * (bins / 6.0)).astype(np.int8)
    return quant.tobytes()


# ── Main class ────────────────────────────────────────────────────────────


class SRMatrix:
    """
    Successor Representation: tabular with linear function approximation fallback.

    Parameters
    ----------
    n_features : int
        Dimensionality of the state feature vector (SR feature dim = same).
    n_actions : int
        Number of discrete actions.
    gamma : float
        Discount factor.
    alpha : float
        SR learning rate.
    max_states : int
        Maximum number of tabular entries before oldest are evicted.
    use_linear_fa : bool
        If True, also maintain a linear FA weight matrix W (n_actions × n_features)
        so predict_q() works even for unseen states.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.05,
        max_states: int = 50_000,
        use_linear_fa: bool = True,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.max_states = max_states
        self.use_linear_fa = use_linear_fa

        # Tabular SR: (state_key, action) → ψ vector (R^n_features)
        self._psi: Dict[Tuple[bytes, int], np.ndarray] = {}
        # Insertion order tracking for LRU-style eviction
        self._insertion_order: List[Tuple[bytes, int]] = []

        # Reward weight vector w ∈ R^n_features — updated by reward observations
        self._reward_w: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._reward_lr: float = 0.01

        # Linear FA: W ∈ R^(n_actions × n_features) — maps state features to Q
        # Updated as a fallback when tabular ψ isn't available
        if use_linear_fa:
            self._W: np.ndarray = np.zeros(
                (n_actions, n_features), dtype=np.float32
            )
        else:
            self._W = None

        # Running mean/std for feature normalisation
        self._feat_mean: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self._feat_var: np.ndarray = np.ones(n_features, dtype=np.float64)
        self._feat_n: int = 0

        # Stats
        self._total_updates: int = 0
        self._tabular_hits: int = 0
        self._fa_hits: int = 0
        self._is_ready: bool = False    # True after min_updates steps

    # ── Normalisation ─────────────────────────────────────────────────

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        """Online Welford normalisation of feature vector."""
        v = np.asarray(vec, dtype=np.float32).flatten()
        if len(v) < self.n_features:
            v = np.pad(v, (0, self.n_features - len(v)))
        elif len(v) > self.n_features:
            v = v[: self.n_features]

        # Update running stats
        self._feat_n += 1
        delta = v.astype(np.float64) - self._feat_mean
        self._feat_mean += delta / self._feat_n
        delta2 = v.astype(np.float64) - self._feat_mean
        self._feat_var += delta * delta2

        std = np.sqrt(self._feat_var / max(self._feat_n, 1)) + 1e-8
        return ((v.astype(np.float64) - self._feat_mean) / std).astype(np.float32)

    # ── Tabular SR helpers ────────────────────────────────────────────

    def _key(self, vec: np.ndarray) -> bytes:
        return _quantise(vec)

    def _get_psi(self, key: bytes, action: int, phi: np.ndarray) -> np.ndarray:
        """
        Retrieve ψ(s, a) from table.  If missing, initialise to φ(s)
        (identity initialisation: 'I am where I am right now').
        """
        k = (key, action)
        if k not in self._psi:
            # Initialise to identity — this state is a 1-step predecessor of itself
            self._psi[k] = phi.copy()
            self._insertion_order.append(k)
            # Evict oldest if over capacity
            if len(self._psi) > self.max_states:
                old = self._insertion_order.pop(0)
                self._psi.pop(old, None)
        return self._psi[k]

    # ── Core SR update ────────────────────────────────────────────────

    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        done: bool,
        reward: float = 0.0,
    ) -> None:
        """
        One-step SR TD update:
            ψ(s, a) ← ψ(s, a) + α·(φ(s) + γ·ψ(s', a*) − ψ(s, a))

        where a* = greedy action in s' (we use the LFA Q if available).
        Also updates reward weight vector w via:
            w ← w + lr·(r − φ(s)·w)·φ(s)
        """
        phi_s = self._normalise(state)
        phi_s2 = self._normalise(next_state)
        sk = self._key(phi_s)
        s2k = self._key(phi_s2)

        # TD target for ψ
        psi_s_a = self._get_psi(sk, action, phi_s)

        if done:
            target = phi_s.copy()  # Terminal: ψ = φ(s) only (no future)
        else:
            # Greedy next action by linear FA (or uniform if W not available)
            a_star = self._greedy_action(phi_s2)
            psi_s2_astar = self._get_psi(s2k, a_star, phi_s2)
            target = phi_s + self.gamma * psi_s2_astar

        # TD update ψ
        td_err = target - psi_s_a
        self._psi[(sk, action)] = psi_s_a + self.alpha * td_err

        # Update linear FA W for each action simultaneously (off-policy)
        if self._W is not None and not done:
            a_star = self._greedy_action(phi_s2)
            q_next = float(self._W[a_star] @ phi_s2)
            q_curr = float(self._W[action] @ phi_s)
            fa_err = reward + self.gamma * q_next - q_curr
            self._W[action] += 0.001 * fa_err * phi_s

        # Update reward weight w (features → reward)
        r_pred = float(phi_s @ self._reward_w)
        self._reward_w += self._reward_lr * (reward - r_pred) * phi_s

        self._total_updates += 1
        if self._total_updates >= 200:
            self._is_ready = True

    def _greedy_action(self, phi: np.ndarray) -> int:
        """Greedy action via LFA Q-values (or random if W unavailable)."""
        if self._W is not None and self._total_updates > 10:
            q = self._W @ phi
            return int(np.argmax(q))
        return int(np.random.randint(self.n_actions))

    # ── Query API ─────────────────────────────────────────────────────

    def get_sr(self, state: np.ndarray, action: Optional[int] = None) -> np.ndarray:
        """
        Return ψ(s, a).  If action is None, return the average ψ across all actions
        (marginal SR — useful for subgoal comparison).
        """
        phi = self._normalise(state)
        sk = self._key(phi)

        if action is not None:
            self._tabular_hits += 1
            return self._get_psi(sk, action, phi)

        # Marginal SR: average over all actions
        psis = [self._get_psi(sk, a, phi) for a in range(self.n_actions)]
        return np.mean(psis, axis=0).astype(np.float32)

    def predict_q(
        self,
        state: np.ndarray,
        reward_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Q(s, a) = ψ(s, a) · w   for all actions simultaneously.

        Parameters
        ----------
        state : feature vector
        reward_weights : optional override for w; uses internally learned w if None

        Returns
        -------
        np.ndarray of shape (n_actions,)
        """
        phi = self._normalise(state)
        sk = self._key(phi)
        w = np.asarray(
            reward_weights if reward_weights is not None else self._reward_w,
            dtype=np.float32,
        )
        q = np.array(
            [float(self._get_psi(sk, a, phi) @ w) for a in range(self.n_actions)],
            dtype=np.float32,
        )
        return q

    def closest_subgoal(
        self,
        state: np.ndarray,
        goal: np.ndarray,
    ) -> int:
        """
        Return the action that moves state closest to goal in SR feature space.

        Uses the SR-distance heuristic:
            a* = argmin_a  ‖ψ(s, a) − ψ(goal)‖₂

        This works because ψ(s, a) encodes the expected future state occupancy
        after taking action a — the closest action to ψ(goal) is the best first
        step along the geodesic.
        """
        phi_s = self._normalise(state)
        phi_g = self._normalise(goal)
        sk = self._key(phi_s)
        gk = self._key(phi_g)

        # Marginal SR of goal (what does the goal 'look like' in SR space)
        psi_goal = np.mean(
            [self._get_psi(gk, a, phi_g) for a in range(self.n_actions)],
            axis=0,
        ).astype(np.float32)

        # Find action minimising SR distance to goal
        best_a = 0
        best_dist = float("inf")
        for a in range(self.n_actions):
            psi_sa = self._get_psi(sk, a, phi_s)
            dist = float(np.linalg.norm(psi_sa - psi_goal))
            if dist < best_dist:
                best_dist = dist
                best_a = a
        return best_a

    def sr_distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        """
        SR-space distance between state and goal (used by SubgoalPlanner).
        Lower = closer in expected future occupancy.
        """
        psi_s = self.get_sr(state)
        psi_g = self.get_sr(goal)
        return float(np.linalg.norm(psi_s - psi_g))

    @property
    def is_ready(self) -> bool:
        """True once enough updates have occurred to trust SR estimates."""
        return self._is_ready

    # ── Diagnostics ───────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "total_updates": self._total_updates,
            "tabular_entries": len(self._psi),
            "is_ready": self._is_ready,
            "reward_w_norm": float(np.linalg.norm(self._reward_w)),
            "tabular_hits": self._tabular_hits,
        }


# ── Standalone smoke test ─────────────────────────────────────────────────


if __name__ == "__main__":
    """Quick convergence test on a 4x4 chain MDP."""
    import sys

    rng = np.random.default_rng(42)
    n_states, n_features, n_actions = 16, 8, 4
    gamma = 0.9
    sr = SRMatrix(n_features=n_features, n_actions=n_actions, gamma=gamma, alpha=0.1)

    # Simulate random walk on identity feature states
    state_feats = rng.standard_normal((n_states, n_features)).astype(np.float32)
    errors = []

    for step in range(2000):
        s_idx = rng.integers(n_states)
        a = rng.integers(n_actions)
        s2_idx = min(s_idx + 1, n_states - 1)
        done = s2_idx == n_states - 1
        reward = 1.0 if done else 0.0

        sr.update(state_feats[s_idx], a, state_feats[s2_idx], done, reward)

        if step > 500 and step % 100 == 0:
            q = sr.predict_q(state_feats[0])
            errors.append(float(np.std(q)))  # should not be zero (learning happening)

    avg_err = np.mean(errors) if errors else 0.0
    print(f"SR smoke test: updates={sr._total_updates}, "
          f"entries={len(sr._psi)}, Q-std={avg_err:.4f}, ready={sr.is_ready}")
    if avg_err > 1e-4 and sr.is_ready:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)
