"""
dyna_system.py
==============
DynaBuffer + ConfirmationMatcher for confirmed Dyna-Q training.

Design
------
Synthetic transitions (WorldModel hallucinations) are stored with a low
weight of SYNTHETIC_WEIGHT (1/11).  When a real ALE transition confirms
a synthetic prediction (cosine similarity >= CONFIRM_THRESHOLD), the
synthetic entry's weight is retroactively boosted to CONFIRMED_WEIGHT (10/11).

Real transitions are always stored at weight 1.0 in the main replay buffer —
this file only manages the *synthetic* side.

Usage in training loop
----------------------
    dyna = DynaSystem(world_model, state_encoder, n_actions)

    # After each real step:
    confirmed = dyna.record_real(ram_before, action, ram_after, game_reward)
    # confirmed: list of DynaSample entries whose predictions matched

    # Run N synthetic steps from current state:
    synth_transitions = dyna.rollout(state_enc, current_x, current_y, n_steps=10)
    # synth_transitions: list of DynaSample — push to replay buffer at .weight

    # Sample from synthetic buffer for additional Q-training:
    batch = dyna.sample_synthetic(batch_size=32)
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

SYNTHETIC_WEIGHT  = 1 / 11     # initial weight for unconfirmed synthetic
CONFIRMED_WEIGHT  = 10 / 11    # weight after real-step confirmation
REAL_WEIGHT       = 1.0        # weight for actual ALE transitions (in main buf)
CONFIRM_THRESHOLD = 0.85       # cosine similarity to count as confirmed
TTL_STEPS         = 500        # discard unconfirmed entries after this many real steps
MAX_DYNA_BUF      = 5_000      # max synthetic entries kept


# ── Data class ─────────────────────────────────────────────────────────────

@dataclass
class DynaSample:
    """One synthetic transition entry."""
    state_enc:   np.ndarray      # compressed state before action
    action:      int
    pred_next:   np.ndarray      # predicted next compressed state
    pred_reward: float
    weight:      float = SYNTHETIC_WEIGHT
    confirmed:   bool  = False
    birth_step:  int   = 0       # real-step count when created
    # Pre-keyed for fast lookup
    state_key:   Optional[bytes] = field(default=None, repr=False)

    def __post_init__(self):
        if self.state_key is None:
            # 8-bit quantized key for fast bucket lookup
            self.state_key = (self.action,
                              tuple((self.state_enc * 255).astype(np.uint8).tolist()))


# ── DynaBuffer ─────────────────────────────────────────────────────────────

class DynaBuffer:
    """
    Stores synthetic transitions with priority weights.
    FIFO-capped at MAX_DYNA_BUF entries.
    """

    def __init__(self) -> None:
        self._buf:   collections.deque = collections.deque(maxlen=MAX_DYNA_BUF)
        self._total_weight = 0.0

    def push(self, sample: DynaSample) -> None:
        if len(self._buf) == self._buf.maxlen:
            evicted = self._buf[0]
            self._total_weight -= evicted.weight
        self._buf.append(sample)
        self._total_weight += sample.weight

    def boost(self, sample: DynaSample) -> None:
        """Retroactively promote a confirmed entry to CONFIRMED_WEIGHT."""
        if not sample.confirmed:
            self._total_weight -= sample.weight
            sample.weight      = CONFIRMED_WEIGHT
            sample.confirmed   = True
            self._total_weight += sample.weight

    def evict_stale(self, current_step: int) -> int:
        """Remove entries older than TTL_STEPS. Returns number evicted."""
        n = 0
        while self._buf and (current_step - self._buf[0].birth_step) > TTL_STEPS:
            evicted = self._buf.popleft()
            self._total_weight -= evicted.weight
            n += 1
        return n

    def sample(self, n: int) -> List[DynaSample]:
        """Weighted sample of n entries (with replacement)."""
        if not self._buf or self._total_weight <= 0:
            return []
        buf = list(self._buf)
        weights = np.array([s.weight for s in buf], dtype=np.float64)
        weights /= weights.sum()
        chosen = np.random.choice(len(buf), size=min(n, len(buf)),
                                  replace=True, p=weights)
        return [buf[i] for i in chosen]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def confirmed_ratio(self) -> float:
        if not self._buf:
            return 0.0
        return sum(1 for s in self._buf if s.confirmed) / len(self._buf)


# ── ConfirmationMatcher ────────────────────────────────────────────────────

class ConfirmationMatcher:
    """
    On each real ALE step, scan the DynaBuffer for synthetic entries whose
    (state, action) and predicted next_state closely match the real transition.
    Confirmed entries get their weight retroactively boosted.
    """

    def __init__(self, dyna_buf: DynaBuffer,
                 threshold: float = CONFIRM_THRESHOLD) -> None:
        self._buf       = dyna_buf
        self._threshold = threshold
        self._n_confirmed  = 0
        self._n_checked    = 0

    def match(
        self,
        real_state_enc: np.ndarray,
        real_action:    int,
        real_next_enc:  np.ndarray,
    ) -> List[DynaSample]:
        """
        Compare real transition to synthetic predictions.
        Returns list of newly confirmed DynaSample entries.
        """
        confirmed = []
        for sample in self._buf._buf:
            if sample.confirmed:
                continue
            if sample.action != real_action:
                continue
            # State match — cosine similarity
            state_sim = _cosine(sample.state_enc, real_state_enc)
            if state_sim < self._threshold:
                continue
            # Next-state match
            next_sim = _cosine(sample.pred_next, real_next_enc)
            self._n_checked += 1
            if next_sim >= self._threshold:
                self._buf.boost(sample)
                confirmed.append(sample)
                self._n_confirmed += 1
        return confirmed

    @property
    def confirmation_rate(self) -> float:
        if self._n_checked == 0:
            return 0.0
        return self._n_confirmed / self._n_checked


# ── DynaSystem ─────────────────────────────────────────────────────────────

class DynaSystem:
    """
    Top-level Dyna-Q manager. Combines buffer + matcher + rollout generation.

    Args:
        world_model:    Object with .predict(state_enc, action) → (next_enc, reward)
        state_encoder:  Object with .encode(ram/255) → Encoded (has .data)
        n_actions:      Action space size
    """

    def __init__(self, world_model, state_encoder, n_actions: int = 18) -> None:
        self._wm        = world_model
        self._enc       = state_encoder
        self._n_actions = n_actions
        self.buf        = DynaBuffer()
        self.matcher    = ConfirmationMatcher(self.buf)
        self._real_step = 0

    def record_real(
        self,
        ram_before:   np.ndarray,
        action:       int,
        ram_after:    np.ndarray,
        game_reward:  float = 0.0,
    ) -> List[DynaSample]:
        """
        Call after every real ALE step.
        Returns confirmed synthetic entries (for logging / buffer boost).
        """
        self._real_step += 1

        s_enc  = self._enc.encode(ram_before.astype(np.float32) / 255.0).data
        ns_enc = self._enc.encode(ram_after.astype(np.float32)  / 255.0).data

        confirmed = self.matcher.match(s_enc, action, ns_enc)
        self.buf.evict_stale(self._real_step)
        return confirmed

    def rollout(
        self,
        start_enc:   np.ndarray,
        current_x:   int,
        current_y:   int,
        n_steps:     int = 10,
        action_mask: Optional[np.ndarray] = None,
        action_boost: Optional[np.ndarray] = None,
    ) -> List[DynaSample]:
        """
        Generate n_steps synthetic transitions from start_enc.
        Falls into known-lethal positions terminate the rollout early
        (no need to simulate all the way to life loss in the WorldModel).

        Args:
            action_mask:   Boolean (n_actions,) from room_constants.
            action_boost:  Float (n_actions,) preference prior.
        """
        from throng4.basal_ganglia.room_constants import (
            is_lethal_zone, is_falling, is_safe_fall, FALL_THRESHOLD,
        )

        samples  = []
        state    = start_enc.copy()
        px, py   = current_x, current_y  # track x,y through rollout

        for step in range(n_steps):
            # Select action — use action_mask + boost if available
            action = self._select_action(state, action_mask, action_boost)

            pred_next, pred_reward = self._wm.predict(state, action)

            # ── Fall / lethal early termination ─────────────────────
            # We can't decode x,y from compressed state perfectly, so we
            # track a rough estimate based on position deltas.
            # If the position was known and this step looks like an unsafe
            # fall → terminate with death penalty instead of continuing.
            lethal = False
            if is_lethal_zone(px, py):
                # Already in the void — every action is fatal
                pred_reward = -5.0
                lethal = True
            # (Falling is detected in real steps via prev_y; in synthetic
            #  we trust the lethal zone check since we track px/py roughly)

            sample = DynaSample(
                state_enc   = state.copy(),
                action      = action,
                pred_next   = pred_next.copy(),
                pred_reward = float(pred_reward),
                birth_step  = self._real_step,
            )
            self.buf.push(sample)
            samples.append(sample)

            if lethal:
                break   # don't continue rollout after lethal position

            state = pred_next

        return samples

    def _select_action(
        self,
        state:        np.ndarray,
        action_mask:  Optional[np.ndarray],
        action_boost: Optional[np.ndarray],
    ) -> int:
        """Sample action proportional to boost (or uniformly if no boost)."""
        probs = np.ones(self._n_actions, dtype=np.float32)
        if action_mask is not None:
            probs[~action_mask] = 0.0
        if action_boost is not None:
            probs = probs + action_boost
            if action_mask is not None:
                probs[~action_mask] = 0.0
        total = probs.sum()
        if total <= 0:
            probs = np.ones(self._n_actions, dtype=np.float32)
            total = self._n_actions
        probs /= total
        return int(np.random.choice(self._n_actions, p=probs))

    @property
    def stats(self) -> dict:
        return {
            "buf_size":          len(self.buf),
            "confirmed_ratio":   round(self.buf.confirmed_ratio, 3),
            "confirmation_rate": round(self.matcher.confirmation_rate, 3),
            "real_steps":        self._real_step,
        }


# ── Helper ─────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
