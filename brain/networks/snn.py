"""
mini_snn.py — 400-neuron Spiking Neural Network for execution context learning.

Fully vectorized implementation — no Python loops over neurons.
All neuron state is stored as numpy arrays; LIF dynamics, homeostasis,
eligibility traces, and Hebbian updates are pure numpy operations.

Performance target: < 0.5ms per process() call at n_neurons=400.

Design rationale
----------------
- Small (400 neurons): can't memorize episodes, only learns statistics
- Hebbian + dopamine: "fire together, wire together" modulated by reward
- Homeostatic threshold: neurons self-regulate firing rate → natural
  novelty detection (unseen states produce irregular, low-rate spikes)
- Small-world topology: local clustering + sparse long-range shortcuts,
  same as biological cortex at this scale

Novelty detection (free property)
----------------------------------
On states the SNN has never seen, neurons haven't calibrated their
thresholds to that input pattern. They fire irregularly and at low rate.
The mean spike rate is therefore low on novel states — which is exactly
the signal we use as context_match. No explicit novelty detector needed.

Robotics safety
---------------
In a real-world deployment, novel physical states (broken hinge, wet floor,
unexpected obstacle) will produce low spike rates → low context_match →
nudge_strength → 0. The main policy runs unassisted. The SNN steps back
rather than confidently applying a learned bias.

Usage
-----
    snn = MiniSNN(input_dim=16, n_neurons=400)

    # After each step (with reward as dopamine):
    context_score = snn.process(state, reward=reward)

    # context_score ∈ [0, 1]:
    #   ~0.0 on novel/unseen states
    #   ~0.3–0.7 on familiar states (homeostatic equilibrium)
    #   consistent across similar states (learned association)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class MiniSNN:
    """
    400-neuron spiking neural network for execution context learning.

    Fully vectorized: all neuron state stored as numpy arrays.
    No Python loops over neurons — LIF dynamics are pure numpy.

    Key properties:
    - Hebbian learning modulated by reward (dopamine signal)
    - Homeostatic threshold → natural novelty detection
    - Small-world topology (local clustering + long-range shortcuts)
    - Weight decay prevents runaway growth
    - Hard cap: nudge contribution never exceeds MAX_NUDGE_STRENGTH

    Args:
        input_dim:       Dimension of compressed state input.
        n_neurons:       Network size. Default 400 — small enough to avoid
                         memorization, large enough for statistical learning.
        connection_prob: Base connection probability for small-world topology.
        learning_rate:   Hebbian learning rate.
        weight_decay:    Per-step weight decay (prevents runaway growth).
        hebbian_interval: How often to run Hebbian update (every N steps).
                         Higher = faster, less frequent learning.
    """

    def __init__(
        self,
        input_dim: int = 16,
        n_neurons: int = 400,
        connection_prob: float = 0.03,
        learning_rate: float = 0.008,
        weight_decay: float = 0.9995,
        hebbian_interval: int = 10,
    ):
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hebbian_interval = hebbian_interval

        # ── Vectorized neuron state arrays ────────────────────────────────────
        # All float32 for speed
        self._activation  = np.zeros(n_neurons, dtype=np.float32)
        self._threshold   = np.full(n_neurons, 0.5, dtype=np.float32)
        self._eligibility = np.zeros(n_neurons, dtype=np.float32)
        self._spikes      = np.zeros(n_neurons, dtype=np.float32)

        # Activity history for homeostatic regulation
        # Shape: (n_neurons, history_len) — ring buffer
        self._history_len = 20
        self._history     = np.zeros((n_neurons, self._history_len), dtype=np.float32)
        self._history_ptr = 0   # Ring buffer write pointer

        # LIF constants
        self._leak_rate        = np.float32(0.9)
        self._homeostatic_rate = np.float32(0.01)
        self._target_rate      = np.float32(0.1)   # 10% firing rate target

        # ── Recurrent weights — small-world topology ──────────────────────────
        self._weights = self._build_small_world(n_neurons, connection_prob)

        # ── Fixed random input projection: input_dim → n_neurons ──────────────
        # Fixed (not learned) so the SNN learns associations, not encoding
        rng = np.random.default_rng(42)
        self._input_proj = rng.uniform(
            0.0, 0.3, (n_neurons, input_dim)
        ).astype(np.float32)

        # ── Running stats ─────────────────────────────────────────────────────
        self._step_count     = 0
        self._mean_spike_ema = np.float32(0.1)   # EMA of mean spike rate

    # ── Build topology ────────────────────────────────────────────────────────

    @staticmethod
    def _build_small_world(n: int, p: float) -> np.ndarray:
        """
        Small-world connectivity: local clusters + sparse long-range shortcuts.
        Fully vectorized — no Python loops.
        """
        # Fibonacci spiral placement (optimal 2D distribution)
        phi = (1 + np.sqrt(5)) / 2
        golden_angle = 2 * np.pi * (1 - 1 / phi)
        idx = np.arange(n)
        r = np.sqrt(idx / max(n, 1))
        theta = idx * golden_angle
        positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Pairwise distances — vectorized
        diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 2)
        distances = np.linalg.norm(diff, axis=-1)              # (n, n)

        weights = np.zeros((n, n), dtype=np.float32)
        local_radius = 0.3

        # Local connections (high probability if close)
        local_mask = (distances < local_radius) & (distances > 0)
        local_fire = local_mask & (np.random.random((n, n)) < p * 5)
        local_w = (np.random.uniform(0.1, 0.5, (n, n)) *
                   (1 - distances / local_radius)).astype(np.float32)
        weights[local_fire] = local_w[local_fire]

        # Long-range shortcuts (sparse)
        long_mask = (distances >= local_radius) & (distances > 0)
        long_fire = long_mask & (np.random.random((n, n)) < 0.003)
        long_w = np.random.uniform(0.05, 0.2, (n, n)).astype(np.float32)
        weights[long_fire] = long_w[long_fire]

        # Symmetrise and zero diagonal
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        return weights

    # ── Forward pass — fully vectorized ──────────────────────────────────────

    def process(self, state: np.ndarray, reward: float = 0.0) -> float:
        """
        Process one state through the SNN. Fully vectorized — no Python loops.

        Steps:
        1. Project state → neuron currents (matrix multiply)
        2. Add recurrent currents from previous spikes (matrix multiply)
        3. LIF update: threshold comparison, reset, leak — all numpy
        4. Homeostatic threshold update — numpy
        5. Eligibility trace update — numpy
        6. Hebbian update every N steps (modulated by reward)
        7. Return mean spike rate as context_match score

        Args:
            state:  Compressed state vector (input_dim,).
            reward: Reward signal used as dopamine for Hebbian modulation.

        Returns:
            context_match ∈ [0, 1]: normalised mean spike rate.
        """
        state_arr = self._ensure_input(state)

        # ── Step 1+2: Compute total input currents ────────────────────────────
        input_currents  = self._input_proj @ state_arr   # (n,)
        recurrent       = self._weights   @ self._spikes  # (n,)
        total_currents  = input_currents + recurrent      # (n,)

        # ── Step 3: LIF dynamics (fully vectorized) ───────────────────────────
        self._activation += total_currents

        # Neurons that fire
        fired = self._activation >= self._threshold      # bool (n,)
        self._spikes = fired.astype(np.float32)

        # Reset fired neurons; leak the rest
        self._activation[fired]  = 0.0
        self._activation[~fired] *= self._leak_rate

        # ── Step 4: Homeostatic threshold regulation ──────────────────────────
        # Write spikes into ring buffer
        self._history[:, self._history_ptr] = self._spikes
        self._history_ptr = (self._history_ptr + 1) % self._history_len

        # Mean firing rate over history window (vectorized over all neurons)
        avg_rate = self._history.mean(axis=1)            # (n,)
        self._threshold += self._homeostatic_rate * (avg_rate - self._target_rate)
        np.clip(self._threshold, 0.1, 1.5, out=self._threshold)

        # ── Step 5: Eligibility trace update ─────────────────────────────────
        self._eligibility = self._eligibility * 0.9 + self._spikes

        # ── Step 6: Hebbian update (every N steps) ───────────────────────────
        if self._step_count % self.hebbian_interval == 0:
            self._hebbian_update(reward)

        self._step_count += 1

        # ── Step 7: Context match score ───────────────────────────────────────
        mean_rate = float(self._spikes.mean())
        self._mean_spike_ema += np.float32(0.01) * (mean_rate - self._mean_spike_ema)
        ema = max(float(self._mean_spike_ema), 0.01)

        raw_score = mean_rate / ema
        return float(np.clip(raw_score, 0.0, 1.0))

    # ── Hebbian learning — vectorized ─────────────────────────────────────────

    def _hebbian_update(self, dopamine: float) -> None:
        """
        Vectorized Hebbian update: fire together, wire together.

        Only updates the active submatrix (neurons with eligibility > 0.1)
        to keep cost proportional to activity, not network size.
        """
        active_idx = np.where(self._eligibility > 0.1)[0]

        if len(active_idx) < 2:
            self._weights *= self.weight_decay
            return

        # Outer product of eligibility traces for active neurons only
        active_elig = self._eligibility[active_idx]
        delta = np.outer(active_elig, active_elig).astype(np.float32)
        np.fill_diagonal(delta, 0)

        # Dopamine modulation
        modulation = float(np.clip(dopamine, -1.0, 1.0))
        if modulation != 0.0:
            delta *= self.learning_rate * modulation
            idx = np.ix_(active_idx, active_idx)
            self._weights[idx] += delta

        # Weight decay + clip (in-place for speed)
        self._weights *= self.weight_decay
        np.clip(self._weights, 0.0, 1.0, out=self._weights)
        np.fill_diagonal(self._weights, 0)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _ensure_input(self, state) -> np.ndarray:
        """Normalise state to float32 array of input_dim."""
        if not isinstance(state, np.ndarray):
            return np.zeros(self.input_dim, dtype=np.float32)
        s = state.astype(np.float32).ravel()
        if s.size < self.input_dim:
            s = np.pad(s, (0, self.input_dim - s.size))
        return s[:self.input_dim]

    def reset_episode(self) -> None:
        """Reset spike/activation state between episodes (keep learned weights)."""
        self._activation[:]  = 0.0
        self._spikes[:]      = 0.0
        self._eligibility[:] = 0.0
        self._history[:]     = 0.0
        self._history_ptr    = 0

    def get_stats(self) -> dict:
        """Diagnostic statistics."""
        return {
            'mean_spike_rate':  float(self._spikes.mean()),
            'mean_eligibility': float(self._eligibility.mean()),
            'mean_threshold':   float(self._threshold.mean()),
            'active_neurons':   int((self._spikes > 0).sum()),
            'weight_density':   float((self._weights > 0).mean()),
            'step_count':       self._step_count,
        }
