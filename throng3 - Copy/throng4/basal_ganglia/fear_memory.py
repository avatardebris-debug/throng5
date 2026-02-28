"""
fear_memory.py — Backward-discounting fear response system.

When the agent dies, this module looks back through the recent trajectory
and applies a decaying "fear signal" to the states/actions that preceded death.
This fear signal creates risk-aversion WITHOUT changing the shaped reward:

  - The replay buffer transitions are NOT modified (pure reward stays clean)
  - Instead, each (state, action) pair in the fear window gets a "fear_weight"
  - The fear_weight is used to BROADEN the MCTS Dirichlet noise (forces more
    exploration away from the fear path) and to REDUCE the Q-update learning
    rate on the specific actions suspected of leading to death.

Think of it as the agent's amygdala tagging recent decisions with a fear
signal that says "this path *might* have caused a bad outcome — be cautious
here, explore alternative routes." It does not say the actions were wrong;
it just dampens their confidence.

Backward discounting formula:
  fear[t] = base_fear * decay^(death_step - t)
  where t is the step index and decay < 1 (closer to death = more fear)

This integrates with:
  - DreamerEngine.mcts: broadens Dirichlet alpha when in a fear-tagged state
  - PortableNNAgent training: reduces lr for fear-tagged (state, action) pairs
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np


@dataclass
class FearEntry:
    """One step in the fear buffer."""
    step:       int          # absolute step number
    x:          int          # player x
    y:          int          # player y
    action:     int          # action taken
    fear:       float        # fear weight [0, 1]


class FearMemory:
    """
    Backward-discounting fear response.

    Usage:
        fear = FearMemory()

        # Each step:
        fear.record(step, x, y, action)

        # On death:
        fear.on_death(step)

        # At planning time — get the fear level for current position:
        level = fear.query(x, y)

    The fear level at a position is the maximum fear weight of all entries
    within a spatial radius of (x, y). This feeds the MCTS Dirichlet noise.
    """

    def __init__(
        self,
        base_fear:    float = 1.0,
        decay:        float = 0.80,
        window:       int   = 60,   # how many steps to look back from death
        spatial_r:    int   = 6,    # x/y radius for spatial fear lookup (pixels)
        max_entries:  int   = 2000, # total fear memory size
    ) -> None:
        self.base_fear  = base_fear
        self.decay      = decay
        self.window     = window
        self.spatial_r  = spatial_r

        # Ring buffer of recent steps (for backward attribution)
        self._ring:  deque[FearEntry] = deque(maxlen=window)

        # Long-term spatial fear map: (x, y) → max fear seen there
        self._map:  Dict[Tuple[int, int], float] = {}
        self._max_entries = max_entries

        # Stats
        self.death_count = 0
        self.total_fear_events = 0

    # ── Public API ──────────────────────────────────────────────────────

    def record(self, step: int, x: int, y: int, action: int) -> None:
        """Record a step into the ring buffer (called every env step)."""
        self._ring.append(FearEntry(step=step, x=x, y=y, action=action, fear=0.0))

    def on_death(self, step: int) -> List[FearEntry]:
        """
        Called when a life is lost.

        Walks backwards through the ring buffer, applying a decaying fear
        signal to each entry. Merges the fear into the spatial map.

        Returns the list of fear-tagged entries (most recent first).
        """
        self.death_count += 1
        tagged: List[FearEntry] = []
        ring = list(self._ring)

        for i, entry in enumerate(reversed(ring)):
            if step - entry.step > self.window:
                break
            # Closer to death (i=0) = highest fear
            fear_val = self.base_fear * (self.decay ** i)
            entry.fear = max(entry.fear, fear_val)

            # Merge into spatial map
            key = (entry.x, entry.y)
            self._map[key] = max(self._map.get(key, 0.0), fear_val)

            tagged.append(entry)
            self.total_fear_events += 1

        # Evict oldest entries if map is too large
        if len(self._map) > self._max_entries:
            oldest = sorted(self._map, key=lambda k: self._map[k])
            for k in oldest[:len(self._map) - self._max_entries]:
                del self._map[k]

        return tagged

    def query(self, x: int, y: int) -> float:
        """
        Return the maximum fear level at or near (x, y).

        A 0.0 means no fear memory near this position.
        A 1.0 means this exact position preceded a death last step.
        """
        max_fear = 0.0
        for (fx, fy), fv in self._map.items():
            if abs(fx - x) <= self.spatial_r and abs(fy - y) <= self.spatial_r:
                max_fear = max(max_fear, fv)
        return max_fear

    def query_action_bias(self, x: int, y: int, n_actions: int) -> np.ndarray:
        """
        Return a per-action uncertainty multiplier based on fear at (x, y).

        Usage in MCTS:
          dirichlet_alpha *= (1.0 + 3.0 * fear_level)  # broaden noise
        Usage in Q-learning:
          lr_multiplier = 1.0 - 0.5 * fear_level        # dampen updates

        Returns np.ndarray of shape (n_actions,) with values in [0, 1],
        where 1.0 = no fear (normal) and lower = more fear.
        """
        fear = self.query(x, y)
        # Uniform dampening across all actions at this position
        # (we don't know which action was "wrong" only that this area is dangerous)
        if fear <= 0.0:
            return np.ones(n_actions, dtype=np.float32)
        return np.full(n_actions, max(0.05, 1.0 - 0.7 * fear), dtype=np.float32)

    def decay_map(self, factor: float = 0.99) -> None:
        """Slowly decay all fear values over time (called once per episode)."""
        for k in list(self._map.keys()):
            self._map[k] *= factor
            if self._map[k] < 0.01:
                del self._map[k]

    def mcts_dirichlet_alpha(self, x: int, y: int, base_alpha: float = 0.3) -> float:
        """
        Return an expanded Dirichlet alpha when in a fear-tagged zone.
        Higher alpha = flatter distribution = more exploration away from the
        standard policy. This naturally broadens search without disabling MCTS.

        Example: base_alpha=0.3, fear=1.0 → alpha=1.2 (4× broader exploration)
        """
        fear = self.query(x, y)
        return base_alpha * (1.0 + 3.0 * fear)

    def summary(self) -> str:
        top_fears = sorted(self._map.items(), key=lambda kv: -kv[1])[:5]
        lines = [
            f"FearMemory: {self.death_count} deaths | "
            f"{self.total_fear_events} fear events | "
            f"{len(self._map)} spatial entries",
        ]
        for (x, y), f in top_fears:
            lines.append(f"  ({x:3d},{y:3d}) fear={f:.3f}")
        return "\n".join(lines)
