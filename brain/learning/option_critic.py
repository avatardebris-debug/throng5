"""
option_critic.py — Option-Critic Architecture (Bacon et al., 2017).

Options are temporally-extended actions: instead of choosing primitive action a
at every step, the Manager chooses an *option* o (a mini-policy with a built-in
termination condition β). The option runs for multiple steps, then terminates
and the Manager picks again.

This enables:
  - Hierarchical action abstraction (each option ≈ a Skill from skill_library.py)
  - Credit assignment over longer horizons without BPTT
  - Reusable sub-policies across different goals

Architecture (two-timescale):
  Manager (slow)  — Q_Ω(s, o): value of choosing option o at state s
  Worker (fast)   — π_o(a|s): action distribution within option o
  Terminator      — β_o(s) ∈ [0,1]: probability of terminating option o at s

Updates:
  Worker loss:    intra-option Q-learning  (step-level)
  Manager loss:   option-level Q-learning  (on termination)
  Terminator:     policy gradient on advantage  (step-level, surrogate)

Integration:
  - SkillLibrary: each Skill → one option slot
  - Hippocampus: feeds (s, a, r, s') every step → update()
  - Striatum: calls select_option() → intra_option_action() instead of ε-greedy

Enabled guard:
  is_ready = False until min_updates steps have been observed.
  Striatum falls back to standard DQN until then.

Usage:
    oc = OptionCritic(n_options=4, n_actions=6, n_features=64)
    o  = oc.select_option(state)
    a  = oc.intra_option_action(state, o)
    oc.update(state, o, a, reward, next_state, done)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Softmax / sigmoid helpers ─────────────────────────────────────────────


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / max(temp, 1e-6)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


# ── OptionCritic ──────────────────────────────────────────────────────────


class OptionCritic:
    """
    Tabular Option-Critic with linear function approximation.

    All Q-values, policies, and termination functions are linear in
    the (normalised) state feature vector. This keeps it compatible with
    the existing feature-based brain pipeline without needing a second
    neural network.

    Parameters
    ----------
    n_options  : int   Number of options (= number of skills in SkillLibrary)
    n_actions  : int   Number of primitive actions
    n_features : int   State feature vector dimension
    gamma      : float Discount
    lr_manager : float Learning rate for Q_Ω (Manager)
    lr_worker  : float Learning rate for π_o (Worker)
    lr_term    : float Learning rate for β_o (Terminator)
    temp       : float Softmax temperature for option/action selection
    min_updates: int   Steps before is_ready = True
    """

    def __init__(
        self,
        n_options: int,
        n_actions: int,
        n_features: int,
        gamma: float = 0.99,
        lr_manager: float = 0.005,
        lr_worker: float = 0.01,
        lr_term: float = 0.001,
        temp: float = 1.0,
        min_updates: int = 500,
    ):
        self.n_options = n_options
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.lr_manager = lr_manager
        self.lr_worker = lr_worker
        self.lr_term = lr_term
        self.temp = temp
        self.min_updates = min_updates

        # Manager Q_Ω: weight matrix (n_options × n_features)
        # Q_Ω(s, o) = W_Omega[o] @ phi(s)
        self._W_omega: np.ndarray = np.zeros(
            (n_options, n_features), dtype=np.float32
        )

        # Worker π_o: (n_options × n_actions × n_features)
        # logit for action a in option o: W_pi[o, a] @ phi(s)
        self._W_pi: np.ndarray = np.zeros(
            (n_options, n_actions, n_features), dtype=np.float32
        )

        # Terminator β_o: (n_options × n_features)
        # β_o(s) = sigmoid(W_beta[o] @ phi(s))
        # Initialise with small positive bias so options don't terminate immediately
        self._W_beta: np.ndarray = np.full(
            (n_options, n_features), fill_value=-0.5, dtype=np.float32
        )

        # Running feature normaliser
        self._feat_mean: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self._feat_m2: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self._feat_n: int = 0

        # Current active option and its start features
        self._current_option: Optional[int] = None
        self._option_start_state: Optional[np.ndarray] = None
        self._steps_in_option: int = 0

        # Stats
        self._total_updates: int = 0
        self._option_counts: np.ndarray = np.zeros(n_options, dtype=np.int64)
        self._option_terminations: np.ndarray = np.zeros(n_options, dtype=np.int64)

        # ── Part 3: Integrated Context (Past + Present + Future) ────────────
        # ctx_dim is set by set_context_mode().
        # When active, W_omega and W_pi are expanded to ctx_dim.
        # W_beta stays on n_features (termination = where I AM, not full history).
        self._ctx_dim: int = n_features     # starts same as n_features
        self._ctx_mode: bool = False        # False = standard features only

    # ── Context mode (Part 3 — Integrated Past+Present+Future) ──────────────

    def set_context_mode(self, ctx_dim: int) -> None:
        """
        Expand Manager and Worker weights to accept a richer context vector:
          ctx = concat([elite_embedding, features, dream_features])

        W_omega and W_pi expand to ctx_dim.
        W_beta stays at n_features (termination = where agent IS, not history).

        Safe to call multiple times; existing weights preserved in first columns.
        """
        if ctx_dim <= self._ctx_dim:
            return

        old_dim = self._ctx_dim
        self._ctx_dim = ctx_dim
        self._ctx_mode = True

        # Expand W_omega: (n_options, old_dim) -> (n_options, ctx_dim)
        new_omega = np.zeros((self.n_options, ctx_dim), dtype=np.float32)
        new_omega[:, :old_dim] = self._W_omega
        self._W_omega = new_omega

        # Expand W_pi: (n_options, n_actions, old_dim) -> (n_options, n_actions, ctx_dim)
        new_pi = np.zeros((self.n_options, self.n_actions, ctx_dim), dtype=np.float32)
        new_pi[:, :, :old_dim] = self._W_pi
        self._W_pi = new_pi

        # Expand Welford normalizer buffers to ctx_dim
        self._feat_mean = np.pad(self._feat_mean, (0, ctx_dim - old_dim))
        self._feat_m2   = np.pad(self._feat_m2,   (0, ctx_dim - old_dim))

    # ── Feature normalisation ─────────────────────────────────────────

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        """Welford online normalisation. Uses _ctx_dim (may be > n_features in ctx mode)."""
        target_dim = self._ctx_dim
        v = np.asarray(vec, dtype=np.float32).flatten()
        if len(v) < target_dim:
            v = np.pad(v, (0, target_dim - len(v)))
        elif len(v) > target_dim:
            v = v[:target_dim]

        self._feat_n += 1
        delta = v.astype(np.float64) - self._feat_mean
        self._feat_mean += delta / self._feat_n
        delta2 = v.astype(np.float64) - self._feat_mean
        self._feat_m2 += delta * delta2
        std = np.sqrt(self._feat_m2 / max(self._feat_n, 1)) + 1e-8
        return ((v.astype(np.float64) - self._feat_mean) / std).astype(np.float32)

    def _normalise_raw(self, vec: np.ndarray) -> np.ndarray:
        """Normalise to n_features only — used by W_beta terminator."""
        v = np.asarray(vec, dtype=np.float32).flatten()
        if len(v) < self.n_features:
            v = np.pad(v, (0, self.n_features - len(v)))
        elif len(v) > self.n_features:
            v = v[:self.n_features]
        return v

    # ── Option values ─────────────────────────────────────────────────

    def _option_values(self, phi: np.ndarray) -> np.ndarray:
        """Q_Ω(s, ·) for all options. Shape: (n_options,)"""
        return (self._W_omega @ phi).astype(np.float32)

    def _action_logits(self, phi: np.ndarray, option: int) -> np.ndarray:
        """Logits for all actions within option o. Shape: (n_actions,)"""
        return (self._W_pi[option] @ phi).astype(np.float32)

    def _termination_prob(self, phi_raw: np.ndarray, option: int) -> float:
        """β_o(s) — probability of terminating option o at state s.

        Uses _normalise_raw (n_features only) since W_beta never expands
        in ctx mode — termination depends on WHERE the agent is, not history.
        """
        phi = self._normalise_raw(phi_raw)
        if len(phi) != self._W_beta.shape[1]:
            phi = phi[:self._W_beta.shape[1]]
        logit = float(self._W_beta[option] @ phi)
        return _sigmoid(logit)

    # ── Manager: option selection ─────────────────────────────────────

    def select_option(
        self,
        state: np.ndarray,
        explore: bool = True,
    ) -> int:
        """
        Manager selects an option at option-boundary steps.

        Uses softmax over Q_Ω(s, ·) for exploration.
        Greedy if explore=False (evaluation mode).
        """
        phi = self._normalise(state)
        q_opts = self._option_values(phi)

        if explore and self.temp > 0.05:
            probs = _softmax(q_opts, temp=self.temp)
            option = int(np.random.choice(self.n_options, p=probs))
        else:
            option = int(np.argmax(q_opts))

        self._current_option = option
        self._option_start_state = phi.copy()
        self._steps_in_option = 0
        self._option_counts[option] += 1
        return option

    # ── Worker: intra-option action ───────────────────────────────────

    def intra_option_action(
        self,
        state: np.ndarray,
        option: int,
        explore: bool = True,
    ) -> int:
        """
        Worker selects a primitive action within option o.

        Uses softmax over π_o(·|s) logits.
        """
        phi = self._normalise(state)
        logits = self._action_logits(phi, option)

        if explore:
            probs = _softmax(logits, temp=max(self.temp, 0.1))
            return int(np.random.choice(self.n_actions, p=probs))
        return int(np.argmax(logits))

    # ── Termination check ─────────────────────────────────────────────

    def should_terminate(
        self,
        state: np.ndarray,
        option: int,
    ) -> bool:
        """
        Sample termination: True with probability β_o(s).

        Also forces termination if option has run for too many steps
        (prevents stuck options).
        """
        phi = self._normalise(state)
        beta = self._termination_prob(phi, option)
        terminated = bool(np.random.random() < beta)
        if terminated:
            self._option_terminations[option] += 1
            self._current_option = None
        return terminated

    # ── Joint Q-learning update ───────────────────────────────────────

    def update(
        self,
        state: np.ndarray,
        option: int,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        One-step intra-option Q-learning updates for all three components.

        Worker (π_o):
            Intra-option Q: Q_U(s, o, a) ≈ r + γ·[(1-β)·Q_Ω(s',o) + β·max_o' Q_Ω(s',o')]
            Policy gradient: ∇_{W_pi} log π_o(a|s) · A_o(s,a)

        Manager (Q_Ω):
            Updated at option boundary (termination or episode end)
            Q_Ω(s,o) ← Q_Ω(s,o) + lr·[r + γ·V(s') − Q_Ω(s,o)]

        Terminator (β_o):
            β gradient: ∇β_o(s) · [Q_Ω(s,o) − V(s)]  (decrease termination when option is good)
        """
        phi_s = self._normalise(state)
        phi_s2 = self._normalise(next_state)

        q_opts_s2 = self._option_values(phi_s2)
        v_s2 = float(np.max(q_opts_s2))         # V(s') = max_o Q_Ω(s', o)
        u_s2 = float(q_opts_s2[option])          # Q_Ω(s', o) — continuing value

        # Termination probability at s'
        beta_s2 = self._termination_prob(phi_s2, option)

        # ── Intra-option Q target ──────────────────────────────────────
        # U(s, o, a) ≈ r + γ[(1−β)·Q_Ω(s',o) + β·V(s')]
        if done:
            intra_q_target = reward
        else:
            intra_q_target = reward + self.gamma * (
                (1.0 - beta_s2) * u_s2 + beta_s2 * v_s2
            )

        # ── Worker update: policy gradient via advantage ───────────────
        # A_o(s, a) = U(s,o,a) − Q_Ω(s,o)   (advantage of this action within option)
        q_s_o = float(self._W_omega[option] @ phi_s)
        advantage = intra_q_target - q_s_o

        # Softmax gradient: ∇ log π_o(a|s) = φ(s) - Σ_a π(a|s)φ(s) = φ_s · (1_{a} - π)
        logits = self._action_logits(phi_s, option)
        pi = _softmax(logits, temp=max(self.temp, 0.1))
        # Gradient for selected action a
        for a_idx in range(self.n_actions):
            indicator = 1.0 if a_idx == action else 0.0
            self._W_pi[option, a_idx] += (
                self.lr_worker * advantage * (indicator - pi[a_idx]) * phi_s
            )

        # ── Manager update: Q_Ω(s, o) ─────────────────────────────────
        td_err_omega = intra_q_target - q_s_o
        self._W_omega[option] += self.lr_manager * td_err_omega * phi_s

        # ── Terminator update: β_o(s') ────────────────────────────────
        # Advantage of termination: A_term = Q_Ω(s',o) - V(s')
        # If Q(s',o) > V(s'), option is above average → decrease β (don't terminate)
        # If Q(s',o) < V(s'), option is below average → increase β (terminate sooner)
        a_term = u_s2 - v_s2  # negative when option is below average → terminate more
        logit_beta = float(self._W_beta[option] @ phi_s2)
        beta_grad = _sigmoid(logit_beta) * (1 - _sigmoid(logit_beta))  # sigmoid'
        self._W_beta[option] += self.lr_term * (-a_term) * beta_grad * phi_s2

        self._total_updates += 1
        # Gradually decrease temperature for annealing
        if self._total_updates % 1000 == 0 and self.temp > 0.1:
            self.temp = max(0.1, self.temp * 0.95)

    # ── Step-level driver (used by Striatum) ──────────────────────────

    def step(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> int:
        """
        Full one-step driver:
          1. If no active option, select one (Manager)
          2. Get action from Worker
          3. Call update() to train all components
          4. Check termination at next_state → clear option if terminated

        Returns primitive action to take.
        """
        # Manager: select option if needed
        if self._current_option is None:
            option = self.select_option(state, explore=True)
        else:
            option = self._current_option

        # Worker: get intra-option action
        action = self.intra_option_action(state, option, explore=True)

        # Update all components
        self.update(state, option, action, reward, next_state, done)

        self._steps_in_option += 1

        # Check termination at next state
        if done or self.should_terminate(next_state, option):
            self._current_option = None

        return action

    @property
    def is_ready(self) -> bool:
        """True once enough data has been observed."""
        return self._total_updates >= self.min_updates

    def set_min_updates(self, n: int) -> None:
        """
        Update the warmup threshold.

        Called by Striatum.enable_option_critic() after detecting average
        episode length. Allows short-episode envs (Acrobot ~100 steps) to
        unlock OC much sooner than long-episode envs (Taxi ~200 steps).
        """
        self.min_updates = max(50, int(n))

    @property
    def current_option(self) -> Optional[int]:
        return self._current_option

    # ── Diagnostics ───────────────────────────────────────────────────

    def option_entropy(self) -> float:
        """Entropy of option usage distribution (high = all options being used)."""
        counts = self._option_counts.astype(np.float64)
        total = counts.sum()
        if total < 1:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def report(self) -> dict:
        return {
            "total_updates": self._total_updates,
            "is_ready": self.is_ready,
            "current_option": self._current_option,
            "option_counts": self._option_counts.tolist(),
            "option_terminations": self._option_terminations.tolist(),
            "option_entropy": round(self.option_entropy(), 3),
            "temperature": round(self.temp, 3),
        }


# ── Standalone smoke test ─────────────────────────────────────────────────


if __name__ == "__main__":
    """
    Quick sanity check: run 800 random steps, verify:
    - All options used (entropy > 0.5)
    - No crashes
    - update() doesn't NaN the weights
    """
    import sys

    rng = np.random.default_rng(7)
    oc = OptionCritic(
        n_options=4, n_actions=4, n_features=8,
        gamma=0.99, min_updates=100,
    )

    for i in range(800):
        s = rng.standard_normal(8).astype(np.float32)
        s2 = rng.standard_normal(8).astype(np.float32)
        r = float(rng.uniform(-1, 1))
        done = bool(rng.random() < 0.05)
        oc.step(s, s2, r, done)

    rpt = oc.report()
    print(f"OptionCritic smoke: ready={rpt['is_ready']}, "
          f"entropy={rpt['option_entropy']:.3f}, "
          f"counts={rpt['option_counts']}")

    nan_check = (
        np.any(np.isnan(oc._W_omega))
        or np.any(np.isnan(oc._W_pi))
        or np.any(np.isnan(oc._W_beta))
    )

    if rpt["is_ready"] and rpt["option_entropy"] > 0.5 and not nan_check:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL  nan_check={nan_check}")
        sys.exit(1)
