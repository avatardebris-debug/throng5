"""
action_chain.py — Store and recall validated action sequences.

Action chains are sequences of actions that successfully navigate
through bottleneck states. They have tiered confidence levels
based on which validation tier confirmed them.

Usage:
    store = ActionChainStore()
    store.store(features, actions=[2, 3, 1, 0, 2], tier="compressed", success_rate=0.65)
    store.promote(state_hash, "worldmodel")

    chain = store.recall(features)
    if chain and chain.tier == "proven":
        execute(chain.actions)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Confidence weights per tier
TIER_WEIGHTS = {
    "compressed": 0.09,     # 1/11 — fast but unreliable
    "worldmodel": 0.90,     # 10× boost when WM confirms
    "real": 9.0,            # Near-certain after real validation
    "proven": 10.0,         # Fully proven and stored
}

TIER_ORDER = ["compressed", "worldmodel", "real", "proven"]


@dataclass
class ActionChain:
    """A validated action sequence for a specific state."""
    state_hash: int
    actions: List[int]
    tier: str = "compressed"
    confidence: float = 0.09
    success_rate: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0
    created_step: int = 0
    last_validated_step: int = 0
    last_used_step: int = 0
    times_used: int = 0
    degradation_count: int = 0

    @property
    def is_proven(self) -> bool:
        return self.tier == "proven"

    @property
    def is_stale(self) -> bool:
        """Chain hasn't been used or validated recently."""
        return self.times_used == 0 and self.degradation_count > 3


class ActionChainStore:
    """
    Hash-indexed store of validated action chains.

    Supports:
    - Store/recall by state similarity
    - Tier promotion (compressed → worldmodel → real → proven)
    - Degradation on re-validation failure
    - Export to Motor Cortex heuristic format
    """

    def __init__(self, max_chains: int = 5000, n_buckets: int = 256):
        self._chains: Dict[int, ActionChain] = {}
        self._max_chains = max_chains
        self._n_buckets = n_buckets

        # Running normalization for hashing
        self._running_mean = np.zeros(8, dtype=np.float64)
        self._running_var = np.ones(8, dtype=np.float64)
        self._n_seen: int = 0

    def _hash_state(self, features: np.ndarray) -> int:
        """Same hashing as BottleneckTracker for consistency."""
        features = np.asarray(features, dtype=np.float32).flatten()
        k = min(8, len(features))
        top_k = features[:k]

        self._n_seen += 1
        alpha = min(0.01, 1.0 / self._n_seen)
        self._running_mean[:k] += alpha * (top_k.astype(np.float64) - self._running_mean[:k])
        diff = top_k.astype(np.float64) - self._running_mean[:k]
        self._running_var[:k] += alpha * (diff ** 2 - self._running_var[:k])

        std = np.sqrt(self._running_var[:k] + 1e-8)
        normalized = (top_k - self._running_mean[:k].astype(np.float32)) / std.astype(np.float32)
        bins = np.clip(((normalized + 3) / 6 * 8).astype(int), 0, 7)
        return hash(bins.tobytes()) % self._n_buckets

    def store(
        self,
        features: np.ndarray,
        actions: List[int],
        tier: str = "compressed",
        success_rate: float = 0.0,
        trials: int = 0,
        step: int = 0,
    ) -> int:
        """
        Store an action chain. Returns state hash.

        If a chain already exists for this state, keeps the higher-tier one.
        """
        h = self._hash_state(features)
        confidence = TIER_WEIGHTS.get(tier, 0.09)

        existing = self._chains.get(h)
        if existing is not None:
            existing_tier_idx = TIER_ORDER.index(existing.tier) if existing.tier in TIER_ORDER else 0
            new_tier_idx = TIER_ORDER.index(tier) if tier in TIER_ORDER else 0
            # Only replace if new tier is higher or success_rate is better at same tier
            if new_tier_idx < existing_tier_idx:
                return h
            if new_tier_idx == existing_tier_idx and success_rate <= existing.success_rate:
                return h

        self._chains[h] = ActionChain(
            state_hash=h,
            actions=list(actions),
            tier=tier,
            confidence=confidence,
            success_rate=success_rate,
            total_trials=trials,
            successful_trials=int(trials * success_rate),
            created_step=step,
            last_validated_step=step,
        )

        self._prune_if_needed()
        return h

    def recall(self, features: np.ndarray) -> Optional[ActionChain]:
        """Find the best chain for this state."""
        h = self._hash_state(features)
        chain = self._chains.get(h)
        if chain is not None:
            chain.times_used += 1
            chain.last_used_step = self._n_seen
        return chain

    def promote(self, state_hash: int, new_tier: str, success_rate: float = 0.0, trials: int = 0) -> bool:
        """Promote a chain to a higher validation tier."""
        if state_hash not in self._chains:
            return False

        chain = self._chains[state_hash]
        tier_idx = TIER_ORDER.index(new_tier) if new_tier in TIER_ORDER else 0
        current_idx = TIER_ORDER.index(chain.tier) if chain.tier in TIER_ORDER else 0

        if tier_idx <= current_idx:
            return False

        chain.tier = new_tier
        chain.confidence = TIER_WEIGHTS.get(new_tier, chain.confidence)
        chain.success_rate = success_rate
        chain.total_trials += trials
        chain.successful_trials += int(trials * success_rate)
        chain.last_validated_step = self._n_seen
        chain.degradation_count = 0
        return True

    def degrade(self, state_hash: int) -> bool:
        """Lower confidence after re-validation failure."""
        if state_hash not in self._chains:
            return False

        chain = self._chains[state_hash]
        chain.degradation_count += 1

        # Drop one tier
        current_idx = TIER_ORDER.index(chain.tier) if chain.tier in TIER_ORDER else 0
        if current_idx > 0:
            chain.tier = TIER_ORDER[current_idx - 1]
            chain.confidence = TIER_WEIGHTS.get(chain.tier, 0.09)

        # If degraded too many times, remove entirely
        if chain.degradation_count > 5:
            del self._chains[state_hash]
            return True

        return True

    def get_proven_chains(self) -> List[ActionChain]:
        """Get all chains at 'proven' tier."""
        return [c for c in self._chains.values() if c.is_proven]

    def export_heuristics(self) -> Dict[int, int]:
        """Export proven chains as state_hash → first_action for Motor Cortex."""
        table = {}
        for chain in self._chains.values():
            if chain.tier in ("real", "proven") and chain.actions:
                table[chain.state_hash] = chain.actions[0]
        return table

    def _prune_if_needed(self) -> None:
        """Remove lowest-confidence chains if over capacity."""
        if len(self._chains) <= self._max_chains:
            return
        sorted_chains = sorted(
            self._chains.items(),
            key=lambda kv: kv[1].confidence * kv[1].success_rate,
        )
        for h, _ in sorted_chains[:len(self._chains) - self._max_chains]:
            del self._chains[h]

    def report(self) -> Dict[str, Any]:
        tier_counts = defaultdict(int)
        for chain in self._chains.values():
            tier_counts[chain.tier] += 1
        return {
            "total_chains": len(self._chains),
            "tiers": dict(tier_counts),
            "proven": tier_counts.get("proven", 0),
            "real": tier_counts.get("real", 0),
            "worldmodel": tier_counts.get("worldmodel", 0),
            "compressed": tier_counts.get("compressed", 0),
        }
