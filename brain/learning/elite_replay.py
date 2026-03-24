"""
elite_replay.py — Top-K Elite Episode Replay Buffer.

Keeps the top-K complete episodes by total reward (human demos, agent bests,
or any labelled source).  During training, samples from these elite episodes
at a rate that STARTS HIGH and asymptotes to a floor — preventing catastrophic
forgetting while allowing the normal replay buffer to dominate later learning.

Concept
-------
* **Elite archive** — ring of K slots, each holding all transitions of one
  complete episode.  When a new episode beats the worst slot its reward, it
  replaces it.  Human demo episodes can be pinned (never evicted).

* **Decay schedule** — elite_fraction(ep) = floor + (start - floor) * exp(-ep / halflife)
  Default: start=0.80, floor=0.20, halflife=200 episodes.
  So ep=0 → 80%, ep=200 → 50%, ep→∞ → 20%.

* **Usage in Striatum** — at each batch sample:
      elite_n  = round(batch_size * elite_fraction)
      normal_n = batch_size - elite_n
  Pull elite_n transitions from the elite archive (uniform across all elites),
  normal_n from the rolling replay buffer, combine and train.

This mirrors published Self-Imitation Learning (SIL, Oh et al 2018) but with
a simpler archive structure and an explicit fraction schedule instead of a
return-advantage gating.

Example
-------
    elite = EliteReplayBuffer(top_k=3, start_fraction=0.80, floor_fraction=0.20)

    # Seed with human demo (pinned = never evicted)
    elite.add_episode(human_transitions, total_reward=500.0, label="human", pinned=True)

    # After each agent episode
    kept = elite.try_add_episode(episode_transitions, total_reward=episode_reward)

    # Inside Striatum.learn() batch loop
    frac = elite.elite_fraction(episode_count)
    elite_batch  = elite.sample(n=round(batch_size * frac))
    normal_batch = replay_buffer.sample(batch_size - len(elite_batch))
    batch = elite_batch + normal_batch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# A single transition in (s, a, r, s', done) format
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class EliteEpisode:
    """One complete episode kept in the archive."""
    transitions: List[Transition]
    total_reward: float
    label: str = "agent"
    pinned: bool = False        # Pinned episodes are never evicted (e.g. human demos)
    episode_idx: int = 0        # Global episode counter when this was added

    def __len__(self) -> int:
        return len(self.transitions)


class EliteReplayBuffer:
    """
    Top-K episode archive with decaying elite fraction schedule.

    Parameters
    ----------
    top_k : int
        Maximum number of elite episodes to keep (default 3).
    start_fraction : float
        Elite sampling fraction at episode 0 (default 0.80 = 80%).
    floor_fraction : float
        Minimum elite fraction after many episodes (default 0.20 = 20%).
    halflife : float
        Number of episodes for fraction to decay halfway from start to floor.
        Default 200: after 200 episodes, elite fraction ≈ 50%.
    """

    def __init__(
        self,
        top_k: int = 3,
        start_fraction: float = 0.80,
        floor_fraction: float = 0.20,
        halflife: float = 200.0,
    ):
        self.top_k = top_k
        self.start_fraction = start_fraction
        self.floor_fraction = floor_fraction
        self.halflife = halflife

        self._archive: List[EliteEpisode] = []  # len <= top_k
        self._total_episodes_seen: int = 0       # for fraction schedule
        self._evictions: int = 0
        self._additions: int = 0

    # ── Archive management ────────────────────────────────────────────

    def add_episode(
        self,
        transitions: List[Transition],
        total_reward: float,
        label: str = "agent",
        pinned: bool = False,
    ) -> bool:
        """
        Force-add an episode (e.g. human demo).  Always accepted regardless
        of reward.  If archive is full, evicts the worst non-pinned slot.

        Returns True if added, False if every slot is pinned and no room.
        """
        ep = EliteEpisode(
            transitions=list(transitions),
            total_reward=total_reward,
            label=label,
            pinned=pinned,
            episode_idx=self._total_episodes_seen,
        )
        if len(self._archive) < self.top_k:
            self._archive.append(ep)
            self._additions += 1
            return True
        # Try to evict worst non-pinned
        evict_idx = self._worst_evictable_idx()
        if evict_idx is None:
            return False  # All slots pinned
        self._archive[evict_idx] = ep
        self._additions += 1
        self._evictions += 1
        return True

    def try_add_episode(
        self,
        transitions: List[Transition],
        total_reward: float,
        label: str = "agent",
    ) -> bool:
        """
        Conditionally add an episode.  Accepted only if it beats the worst
        evictable slot's reward (or archive isn't full yet).

        Returns True if the episode was kept (is elite).
        """
        self._total_episodes_seen += 1

        ep = EliteEpisode(
            transitions=list(transitions),
            total_reward=total_reward,
            label=label,
            pinned=False,
            episode_idx=self._total_episodes_seen,
        )

        if len(self._archive) < self.top_k:
            self._archive.append(ep)
            self._additions += 1
            return True

        worst_idx = self._worst_evictable_idx()
        if worst_idx is None:
            return False  # All slots pinned

        if total_reward > self._archive[worst_idx].total_reward:
            self._archive[worst_idx] = ep
            self._additions += 1
            self._evictions += 1
            return True

        return False

    def _worst_evictable_idx(self) -> Optional[int]:
        """Index of the lowest-reward non-pinned slot, or None if all pinned."""
        worst_idx = None
        worst_reward = float("inf")
        for i, ep in enumerate(self._archive):
            if not ep.pinned and ep.total_reward < worst_reward:
                worst_reward = ep.total_reward
                worst_idx = i
        return worst_idx

    # ── Sampling ──────────────────────────────────────────────────────

    def sample(self, n: int) -> List[Transition]:
        """
        Sample n transitions uniformly from across all elite episodes.

        Sampling is uniform within the combined pool of all elite transitions.
        This means longer episodes are proportionally more represented —
        which is correct, since longer high-reward episodes have more
        informative transitions.
        """
        if not self._archive or n <= 0:
            return []

        # Build flat pool (concatenate all elite transitions)
        # We use index arithmetic to avoid actually concatenating arrays
        pool_sizes = [len(ep) for ep in self._archive]
        total = sum(pool_sizes)
        if total == 0:
            return []

        n = min(n, total)

        # Uniform random indices into flat pool
        flat_indices = np.random.randint(0, total, size=n)
        transitions: List[Transition] = []

        # Map flat_idx → (episode_idx, transition_idx)
        cumulative = np.cumsum([0] + pool_sizes)
        for flat_idx in flat_indices:
            ep_i = int(np.searchsorted(cumulative, flat_idx, side="right")) - 1
            ep_i = min(ep_i, len(self._archive) - 1)
            t_i = flat_idx - cumulative[ep_i]
            t_i = min(t_i, len(self._archive[ep_i].transitions) - 1)
            transitions.append(self._archive[ep_i].transitions[int(t_i)])

        return transitions

    def sample_by_reward(self, n: int) -> List[Transition]:
        """
        Sample n transitions weighted by episode total_reward.
        Gives higher-reward elite episodes proportionally more weight.
        """
        if not self._archive or n <= 0:
            return []

        rewards = np.array([ep.total_reward for ep in self._archive], dtype=np.float64)
        # Shift to positive if any negatives (e.g. reward=-200)
        rewards -= rewards.min() - 1e-3
        weights = rewards / rewards.sum()

        pool_sizes = [len(ep) for ep in self._archive]
        total = sum(pool_sizes)
        if total == 0:
            return []

        n = min(n, total)
        transitions: List[Transition] = []

        # Weight episode selection by reward, then uniform within episode
        ep_counts = np.round(weights * n).astype(int)
        # Fix rounding to hit exactly n
        ep_counts[-1] += n - ep_counts.sum()

        for ep_i, count in enumerate(ep_counts):
            ep = self._archive[ep_i]
            if count <= 0 or not ep.transitions:
                continue
            pool = ep.transitions
            k = min(count, len(pool))
            indices = np.random.randint(0, len(pool), size=k)
            transitions.extend(pool[i] for i in indices)

        return transitions[:n]

    # ── Fraction schedule ─────────────────────────────────────────────

    def elite_fraction(self, episode: Optional[int] = None) -> float:
        """
        Compute current elite sampling fraction.

        fraction(ep) = floor + (start - floor) * exp(-episode / halflife * ln2)

        This gives:
          ep=0        → start_fraction  (e.g. 80%)
          ep=halflife → midpoint        (e.g. 50%)
          ep→∞        → floor_fraction  (e.g. 20%)
        """
        if episode is None:
            episode = self._total_episodes_seen
        if self.halflife <= 0:
            return self.floor_fraction
        span = self.start_fraction - self.floor_fraction
        decay = math.exp(-episode * math.log(2) / self.halflife)
        return self.floor_fraction + span * decay

    # ── Status ────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return len(self._archive) == 0

    @property
    def size(self) -> int:
        return len(self._archive)

    @property
    def min_reward(self) -> float:
        if not self._archive:
            return float("-inf")
        return min(ep.total_reward for ep in self._archive)

    @property
    def max_reward(self) -> float:
        if not self._archive:
            return float("-inf")
        return max(ep.total_reward for ep in self._archive)

    # ── Context embedding (Phase 5 Part 3) ───────────────────────────

    def summary_embedding(self, dim: int = 8) -> np.ndarray:
        """
        Return a fixed-size embedding summarising the elite archive.

        Computed as the mean state vector across all elite transitions,
        then projected to `dim` via a deterministic random projection
        (no sklearn or extra dependencies).

        Used by Striatum._build_ctx() to fill the 'past memory' slot
        of the integrated past+present+future context fed to OptionCritic.

        Returns zeros if archive is empty.
        """
        if not self._archive:
            return np.zeros(dim, dtype=np.float32)

        # Collect mean state from each elite episode
        state_means = []
        for ep in self._archive:
            if not ep.transitions:
                continue
            states = np.array([t[0] for t in ep.transitions], dtype=np.float32)
            state_means.append(states.mean(axis=0))

        if not state_means:
            return np.zeros(dim, dtype=np.float32)

        # Grand mean across all elite episodes
        grand_mean = np.mean(state_means, axis=0).astype(np.float32)
        d = len(grand_mean)

        if d <= dim:
            # Pad to dim if state is smaller
            return np.pad(grand_mean, (0, dim - d))

        # Random projection: stable hash-seeded matrix (reproducible, no training)
        rng = np.random.default_rng(seed=42)
        proj = rng.standard_normal((d, dim)).astype(np.float32) / np.sqrt(dim)
        emb = grand_mean @ proj       # (dim,)

        # L2 normalise so magnitude doesn't swamp the other ctx components
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def report(self) -> Dict[str, Any]:
        """Status summary for telemetry."""
        current_frac = self.elite_fraction()
        slots = [
            {
                "label": ep.label,
                "reward": round(ep.total_reward, 2),
                "steps": len(ep),
                "pinned": ep.pinned,
            }
            for ep in self._archive
        ]
        return {
            "slots": slots,
            "top_k": self.top_k,
            "filled": len(self._archive),
            "total_episodes_seen": self._total_episodes_seen,
            "current_elite_fraction": round(current_frac, 3),
            "additions": self._additions,
            "evictions": self._evictions,
        }

    def __repr__(self) -> str:
        rewards = [round(ep.total_reward, 1) for ep in self._archive]
        frac = self.elite_fraction()
        return (
            f"EliteReplayBuffer(top_k={self.top_k}, "
            f"filled={len(self._archive)}, "
            f"rewards={rewards}, "
            f"elite_frac={frac:.2f})"
        )
