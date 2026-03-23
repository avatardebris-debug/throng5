"""
go_explore.py — Go-Explore state archive for hard-exploration environments.

Based on Ecoffet et al. 2021 (Go-Explore).

Core idea
---------
Instead of random exploration, maintain a cell-based archive of "interesting
states" — states that were novel, rarely visited, or had high predicted value.
When exploration is ineffective, RETURN to a promising archived state and
explore outward from there.

This transforms exploration from:
  "random walk from start" → "targeted walk from known interesting states"

Especially effective for:
  • FrozenLake       (most episodes reach the same dead-ends repeatedly)
  • MountainCar      (never reaches the top via random exploration from start)
  • Montezuma's Revenge (dense maze, sparse reward)
  • Any env where the goal is rarely reached from the initial state

Cell resolution
---------------
States are discretized into cells using rounded+hashed features. Two states
map to the same cell if they are "close enough". Cell resolution is tunable
per-environment:
  coarse  (0.25): for high-dim states (84-dim abstract features)
  medium  (0.10): for small continuous state spaces (CartPole, MountainCar)
  fine    (0.05): for environments where small differences matter

Archive eviction
----------------
Archive is bounded (default 5000 cells). When full, evicts the cell with:
  lowest visit_count * best_reward_seen (least productive cells go first)

Usage
-----
    archive = GoExploreArchive(max_cells=5000, resolution=0.10)

    # Every step:
    archive.add_state(state, reward=reward, step=step_count)

    # On sparse episode (to get a good starting point for next ep):
    good_state, good_idx = archive.select_return_state()
    # good_state can be used to restore env position or as a "dream start"

    # Report
    archive.report()  → dict of stats
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GoExploreCell:
    """One cell in the archive. Tracks visit count and best reward seen."""

    __slots__ = ("state", "visits", "best_reward", "best_step", "cell_key")

    def __init__(self, state: np.ndarray, reward: float, step: int, key: str):
        self.state = state.copy()
        self.visits = 1
        self.best_reward = reward
        self.best_step = step
        self.cell_key = key

    def update(self, state: np.ndarray, reward: float, step: int) -> None:
        self.visits += 1
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_step = step
            self.state = state.copy()

    @property
    def score(self) -> float:
        """Selection score: prefer rarely-visited, high-reward cells."""
        # Combine novelty (1/visits) + reward signal
        novelty = 1.0 / (self.visits + 1)
        reward_bonus = max(0.0, self.best_reward)
        return novelty + 0.5 * reward_bonus


class GoExploreArchive:
    """
    Cell-based exploration archive.

    Parameters
    ----------
    max_cells : int
        Maximum cells to keep. When full, evicts least productive cells.
    resolution : float
        Discretization step for cell hashing (0.10 recommended for most envs).
    min_steps_before_return : int
        Minimum episode steps before suggesting a return state.
        Avoids returning to very early states that haven't been explored.
    """

    def __init__(
        self,
        max_cells: int = 5000,
        resolution: float = 0.10,
        min_steps_before_return: int = 5,
    ):
        self.max_cells = max_cells
        self.resolution = resolution
        self.min_steps_before_return = min_steps_before_return

        self._cells: Dict[str, GoExploreCell] = {}
        self._total_visits = 0
        self._total_evictions = 0
        self._states_seen = 0

    # ── Core API ──────────────────────────────────────────────────────

    def add_state(
        self,
        state: np.ndarray,
        reward: float = 0.0,
        step: int = 0,
    ) -> str:
        """
        Record a state visit. Returns the cell key.

        This should be called every step during an episode.
        """
        self._states_seen += 1
        self._total_visits += 1

        key = self._cell_key(state)

        if key in self._cells:
            self._cells[key].update(state, reward, step)
        else:
            if len(self._cells) >= self.max_cells:
                self._evict()
            self._cells[key] = GoExploreCell(state, reward, step, key)

        return key

    def select_return_state(
        self,
        strategy: str = "mixed",
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Select a promising state to return to for the next episode.

        Strategies
        ----------
        "novel"   : highest novelty (least visited cells)
        "best"    : highest best_reward seen
        "mixed"   : score = 1/(visits+1) + 0.5 * best_reward  (default)
        "random"  : random cell (for comparison)

        Returns
        -------
        (state_array, cell_key) or (None, None) if archive empty.
        """
        if not self._cells:
            return None, None

        # Filter: only cells with enough visits (not just start-state noise)
        candidates = [
            c for c in self._cells.values()
            if c.best_step >= self.min_steps_before_return
        ]
        if not candidates:
            candidates = list(self._cells.values())

        if strategy == "novel":
            best = min(candidates, key=lambda c: c.visits)
        elif strategy == "best":
            best = max(candidates, key=lambda c: c.best_reward)
        elif strategy == "random":
            best = np.random.choice(candidates)
        else:  # "mixed"
            # Softmax sample over scores (stochastic, not pure greedy)
            scores = np.array([c.score for c in candidates], dtype=np.float64)
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            idx = np.random.choice(len(candidates), p=probs)
            best = candidates[idx]

        return best.state, best.cell_key

    def get_top_states(self, n: int = 5) -> List[GoExploreCell]:
        """Return the top-n cells by score."""
        cells = sorted(self._cells.values(), key=lambda c: c.score, reverse=True)
        return cells[:n]

    # ── Internals ─────────────────────────────────────────────────────

    def _cell_key(self, state: np.ndarray) -> str:
        """
        Hash a state into a discrete cell key.

        Discretize by resolution → round → hash as bytes.
        Robust to float noise and works for any state dimensionality.
        """
        quantized = np.round(
            np.asarray(state, dtype=np.float32) / self.resolution
        ).astype(np.int16)
        # Use first 16 dims max to keep key short (prevents hash collision storms
        # in very high-dim spaces — abstract_features already compresses to 84)
        key_arr = quantized[:16] if len(quantized) > 16 else quantized
        return hashlib.md5(key_arr.tobytes()).hexdigest()[:12]

    def _evict(self) -> None:
        """Evict the least productive cell (lowest score)."""
        if not self._cells:
            return
        worst_key = min(self._cells, key=lambda k: self._cells[k].score)
        del self._cells[worst_key]
        self._total_evictions += 1

    def reset_episode(self) -> None:
        """Call at start of each episode to reset per-episode counters."""
        pass   # Archive persists across episodes — that's the point

    # ── Stats ─────────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        if not self._cells:
            return {"cells": 0, "visits": 0}

        scores = [c.score for c in self._cells.values()]
        rewards = [c.best_reward for c in self._cells.values()]
        visits = [c.visits for c in self._cells.values()]

        return {
            "cells": len(self._cells),
            "max_cells": self.max_cells,
            "total_states_seen": self._states_seen,
            "total_evictions": self._total_evictions,
            "avg_visits_per_cell": round(float(np.mean(visits)), 2),
            "max_visits_single_cell": int(max(visits)),
            "best_reward_in_archive": round(float(max(rewards)), 3),
            "avg_score": round(float(np.mean(scores)), 4),
        }

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return (
            f"GoExploreArchive(cells={len(self._cells)}, "
            f"seen={self._states_seen}, evictions={self._total_evictions})"
        )
