"""
reward_discovery.py — Intrinsic reward from RAM state changes.

Games like Montezuma and Lolo give sparse rewards (+100 for key, +0 for
everything else). This module generates DENSE intrinsic rewards by
tracking meaningful RAM changes:

  - Player moved closer to an item → small positive
  - New RAM byte changed for the first time → novelty bonus
  - Subgoal-correlated bytes changed → progress bonus
  - Death-correlated bytes triggered → early warning penalty
  - Count-based exploration → reward for visiting rare states

Works with RAMSemanticMapper to know WHICH bytes are meaningful.

Usage:
    discoverer = RewardDiscovery(ram_size=128)
    intrinsic_r = discoverer.compute(ram_before, ram_after, action, extrinsic_reward)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class RewardDiscovery:
    """
    Generates intrinsic rewards from RAM state changes.

    Provides dense reward signal in sparse-reward games by identifying
    meaningful state transitions that indicate progress, novelty, or danger.
    """

    def __init__(
        self,
        ram_size: int = 128,
        novelty_scale: float = 0.5,
        progress_scale: float = 1.0,
        proximity_scale: float = 0.1,
        danger_scale: float = -0.3,
        decay_rate: float = 0.999,
    ):
        self._ram_size = ram_size

        # Reward scales
        self._novelty_scale = novelty_scale
        self._progress_scale = progress_scale
        self._proximity_scale = proximity_scale
        self._danger_scale = danger_scale
        self._decay_rate = decay_rate

        # == Novelty tracking (count-based) ==
        # Visit counts per RAM state hash
        self._state_visits: Dict[int, int] = defaultdict(int)
        self._byte_first_change: Set[Tuple[int, int]] = set()  # (addr, value)

        # == Progress tracking ==
        # Bytes known to correlate with reward (from RAMSemanticMapper)
        self._subgoal_bytes: Set[int] = set()
        self._death_bytes: Set[int] = set()
        self._position_bytes: List[int] = []  # ordered pairs: [px, py, ...]

        # == Target tracking (for proximity reward) ==
        self._target_positions: List[Tuple[int, int]] = []  # (x_addr, y_addr)
        self._item_positions: Dict[str, Tuple[int, int]] = {}  # name → (x, y)

        # == Running statistics ==
        self._total_steps: int = 0
        self._total_intrinsic: float = 0.0
        self._max_intrinsic: float = 0.0
        self._reward_history: List[float] = []

    # ── Configuration (from RAMSemanticMapper) ───────────────────────

    def configure_from_mapper(self, mapper) -> None:
        """
        Auto-configure from a trained RAMSemanticMapper.

        Extracts which bytes are subgoal-relevant, death-relevant,
        and position-tracking.
        """
        # Subgoal bytes — change at reward moments
        for item in mapper.get_subgoal_bytes():
            self._subgoal_bytes.add(item["addr"])

        # Death bytes — change at death
        for addr, prof in mapper._profiles.items():
            if prof.changed_at_death > 2:
                self._death_bytes.add(addr)

        # Position bytes — high-frequency changers
        registry = mapper.get_registry()
        for entry in registry.get("position", []):
            self._position_bytes.append(entry["addr"])

        # Entity groups — for proximity to items
        for group in mapper.get_entity_groups():
            if len(group["bytes"]) >= 2:
                self._target_positions.append(
                    (group["bytes"][0], group["bytes"][1])
                )

    def configure_manual(
        self,
        subgoal_bytes: Optional[List[int]] = None,
        death_bytes: Optional[List[int]] = None,
        position_bytes: Optional[List[int]] = None,
        item_positions: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        """Manually configure known RAM addresses."""
        if subgoal_bytes:
            self._subgoal_bytes.update(subgoal_bytes)
        if death_bytes:
            self._death_bytes.update(death_bytes)
        if position_bytes:
            self._position_bytes = position_bytes
        if item_positions:
            self._item_positions.update(item_positions)

    # ── Core Reward Computation ──────────────────────────────────────

    def compute(
        self,
        ram_before: np.ndarray,
        ram_after: np.ndarray,
        action: int = 0,
        extrinsic_reward: float = 0.0,
        done: bool = False,
    ) -> float:
        """
        Compute intrinsic reward from a RAM transition.

        Returns a float combining:
          - Novelty bonus (visiting rare states)
          - Progress bonus (subgoal bytes changed)
          - Proximity bonus (closer to known items)
          - Danger penalty (death-correlated changes)

        The intrinsic reward is added to the extrinsic reward
        by the caller.
        """
        self._total_steps += 1
        r_before = np.asarray(ram_before, dtype=np.uint8).flatten()[:self._ram_size]
        r_after = np.asarray(ram_after, dtype=np.uint8).flatten()[:self._ram_size]

        intrinsic = 0.0

        # 1. Novelty: count-based exploration bonus
        intrinsic += self._novelty_reward(r_after)

        # 2. Progress: subgoal-correlated byte changes
        intrinsic += self._progress_reward(r_before, r_after)

        # 3. Proximity: getting closer to known items/targets
        intrinsic += self._proximity_reward(r_before, r_after)

        # 4. Danger: early warning from death-correlated changes
        intrinsic += self._danger_reward(r_before, r_after, done)

        # 5. First-time byte values: novelty for new discoveries
        intrinsic += self._first_change_reward(r_before, r_after)

        # Decay intrinsic over time (avoid over-reliance)
        time_factor = self._decay_rate ** (self._total_steps / 1000.0)
        intrinsic *= time_factor

        # Track
        self._total_intrinsic += intrinsic
        self._max_intrinsic = max(self._max_intrinsic, abs(intrinsic))
        if len(self._reward_history) < 10000:
            self._reward_history.append(intrinsic)

        return intrinsic

    def _novelty_reward(self, ram: np.ndarray) -> float:
        """Count-based exploration: reward rarely-visited states."""
        state_hash = hash(ram.tobytes()) % 100000
        self._state_visits[state_hash] += 1
        count = self._state_visits[state_hash]

        # 1/sqrt(N) bonus — high for first visit, decaying
        bonus = self._novelty_scale / (count ** 0.5)
        return bonus

    def _progress_reward(self, before: np.ndarray, after: np.ndarray) -> float:
        """Reward changes in subgoal-correlated bytes."""
        if not self._subgoal_bytes:
            return 0.0

        reward = 0.0
        for addr in self._subgoal_bytes:
            if addr < len(before) and addr < len(after):
                if before[addr] != after[addr]:
                    # Subgoal byte changed — strong positive signal
                    # Direction matters: 0→1 is usually progress
                    if after[addr] > before[addr]:
                        reward += self._progress_scale
                    else:
                        # Regression (e.g., lives decreased) — mild negative
                        reward -= self._progress_scale * 0.3
        return reward

    def _proximity_reward(self, before: np.ndarray, after: np.ndarray) -> float:
        """Reward getting closer to known targets."""
        if not self._position_bytes or len(self._position_bytes) < 2:
            return 0.0
        if not self._target_positions:
            return 0.0

        reward = 0.0
        # Player position (first two position bytes)
        px_addr = self._position_bytes[0]
        py_addr = self._position_bytes[1] if len(self._position_bytes) > 1 else px_addr

        if px_addr >= len(before) or py_addr >= len(before):
            return 0.0

        player_x_before = float(before[px_addr])
        player_y_before = float(before[py_addr])
        player_x_after = float(after[px_addr])
        player_y_after = float(after[py_addr])

        # Reward for each target
        for tx_addr, ty_addr in self._target_positions:
            if tx_addr >= len(after) or ty_addr >= len(after):
                continue
            target_x = float(after[tx_addr])
            target_y = float(after[ty_addr])

            dist_before = ((player_x_before - target_x)**2 + (player_y_before - target_y)**2)**0.5
            dist_after = ((player_x_after - target_x)**2 + (player_y_after - target_y)**2)**0.5

            # Reward for getting closer, penalty for moving away
            delta = dist_before - dist_after  # Positive = got closer
            reward += delta * self._proximity_scale / max(dist_before, 1.0)

        return reward

    def _danger_reward(
        self, before: np.ndarray, after: np.ndarray, done: bool,
    ) -> float:
        """Penalize changes in death-correlated bytes."""
        if not self._death_bytes:
            return 0.0

        penalty = 0.0
        for addr in self._death_bytes:
            if addr < len(before) and addr < len(after):
                if before[addr] != after[addr]:
                    penalty += self._danger_scale

        # Extra penalty if episode ended (actual death)
        if done and penalty < 0:
            penalty *= 2.0

        return penalty

    def _first_change_reward(self, before: np.ndarray, after: np.ndarray) -> float:
        """Novelty bonus for byte values never seen before."""
        bonus = 0.0
        for addr in range(min(len(before), len(after))):
            if before[addr] != after[addr]:
                key = (addr, int(after[addr]))
                if key not in self._byte_first_change:
                    self._byte_first_change.add(key)
                    bonus += self._novelty_scale * 0.1  # Small per-byte bonus
        return bonus

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        avg = self._total_intrinsic / max(self._total_steps, 1)
        return {
            "total_steps": self._total_steps,
            "total_intrinsic": round(self._total_intrinsic, 4),
            "avg_intrinsic": round(avg, 6),
            "max_intrinsic": round(self._max_intrinsic, 4),
            "unique_states": len(self._state_visits),
            "byte_discoveries": len(self._byte_first_change),
            "configured": {
                "subgoal_bytes": len(self._subgoal_bytes),
                "death_bytes": len(self._death_bytes),
                "position_bytes": len(self._position_bytes),
                "targets": len(self._target_positions),
            },
        }
