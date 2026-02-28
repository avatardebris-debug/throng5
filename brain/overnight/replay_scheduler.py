"""
replay_scheduler.py — Priority-based replay scheduling for overnight consolidation.

Decides WHAT to replay and in what ORDER during the overnight loop:
  1. High-surprise transitions (large TD error)
  2. Edge cases (near-death recoveries, first-time discoveries)
  3. Reward-rich episodes (successful strategies to reinforce)
  4. Staleness-weighted sampling (old memories decay unless refreshed)

Used by the DreamLoop to feed batches to each brain region for
offline training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReplayPriority:
    """Priority metadata for a single transition."""
    idx: int
    td_error: float = 0.0
    reward: float = 0.0
    is_edge_case: bool = False
    is_discovery: bool = False
    staleness: int = 0        # Steps since last replayed
    times_replayed: int = 0

    @property
    def score(self) -> float:
        """Combined priority score."""
        s = abs(self.td_error) * 2.0
        s += abs(self.reward) * 1.5
        if self.is_edge_case:
            s += 3.0
        if self.is_discovery:
            s += 2.0
        # Staleness bonus: older unreplayed items get higher priority
        s += min(self.staleness * 0.01, 2.0)
        # Diminishing returns on re-replay
        s *= 1.0 / (1.0 + self.times_replayed * 0.3)
        return s


class ReplayScheduler:
    """
    Schedules replay batches for the overnight dream loop.

    Maintains a priority queue of transitions and produces
    batches sorted by replay priority (surprise × reward × staleness).
    """

    def __init__(self, max_items: int = 50000):
        self._priorities: List[ReplayPriority] = []
        self._max_items = max_items
        self._total_scheduled = 0

    def add(
        self,
        idx: int,
        td_error: float = 0.0,
        reward: float = 0.0,
        is_edge_case: bool = False,
        is_discovery: bool = False,
    ) -> None:
        """Add a transition to the replay schedule."""
        self._priorities.append(ReplayPriority(
            idx=idx, td_error=td_error, reward=reward,
            is_edge_case=is_edge_case, is_discovery=is_discovery,
        ))
        if len(self._priorities) > self._max_items:
            # Remove lowest-priority items
            self._priorities.sort(key=lambda p: p.score, reverse=True)
            self._priorities = self._priorities[:self._max_items]

    def age_all(self) -> None:
        """Increment staleness for all items (called each overnight cycle)."""
        for p in self._priorities:
            p.staleness += 1

    def schedule_batch(self, batch_size: int = 64) -> List[int]:
        """
        Produce the next batch of transition indices to replay.

        Returns indices sorted by priority (highest first).
        Marks replayed items to reduce their future priority.
        """
        if not self._priorities:
            return []

        # Sort by score descending
        self._priorities.sort(key=lambda p: p.score, reverse=True)

        # Take top batch
        batch = self._priorities[:batch_size]
        indices = [p.idx for p in batch]

        # Mark as replayed
        for p in batch:
            p.times_replayed += 1
            p.staleness = 0

        self._total_scheduled += len(indices)
        return indices

    def schedule_edge_cases(self, n: int = 10) -> List[int]:
        """Schedule specifically edge-case transitions for targeted replay."""
        edge = [p for p in self._priorities if p.is_edge_case]
        edge.sort(key=lambda p: p.score, reverse=True)
        indices = [p.idx for p in edge[:n]]
        for p in edge[:n]:
            p.times_replayed += 1
        return indices

    def update_td_error(self, idx: int, new_td: float) -> None:
        """Update TD error for a transition (after re-evaluation)."""
        for p in self._priorities:
            if p.idx == idx:
                p.td_error = new_td
                break

    def stats(self) -> Dict[str, Any]:
        if not self._priorities:
            return {"n_items": 0, "total_scheduled": self._total_scheduled}
        scores = [p.score for p in self._priorities]
        return {
            "n_items": len(self._priorities),
            "total_scheduled": self._total_scheduled,
            "avg_score": round(np.mean(scores), 3),
            "max_score": round(np.max(scores), 3),
            "edge_cases": sum(1 for p in self._priorities if p.is_edge_case),
            "discoveries": sum(1 for p in self._priorities if p.is_discovery),
        }
