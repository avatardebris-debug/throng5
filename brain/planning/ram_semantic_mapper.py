"""
ram_semantic_mapper.py — Auto-discover game objects from RAM diffs.

Watches RAM changes over time and classifies bytes into semantic
categories:
  - Position:  Changes nearly every frame (player/enemy coords)
  - Counter:   Increments/decrements steadily (score, timer)
  - State:     Changes rarely, often binary (inventory, door state)
  - Subgoal:   Changes that correlate with reward events

Can consume HumanRecorder data or learn online during agent play.

Usage:
    mapper = RAMSemanticMapper()
    mapper.observe(ram_before, ram_after, action, reward, done)
    # ... many observations later ...
    registry = mapper.get_registry()
    # {'position': [(0x31, 'player_x?'), ...], 'state': [(0xC5, 'key?'), ...]}
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class RAMByteProfile:
    """Statistical profile of a single RAM byte."""

    def __init__(self, address: int):
        self.address = address
        self.change_count: int = 0
        self.total_observations: int = 0

        # Value tracking
        self.values_seen: Set[int] = set()
        self.last_value: int = 0
        self.min_value: int = 255
        self.max_value: int = 0

        # Change pattern
        self.increments: int = 0       # Value increased
        self.decrements: int = 0       # Value decreased
        self.resets: int = 0           # Value jumped to 0
        self.flips: int = 0            # Binary-like changes (0↔non-zero)

        # Correlation with events
        self.changed_at_reward: int = 0
        self.changed_at_death: int = 0
        self.changed_at_episode_start: int = 0
        self.reward_total_when_changed: float = 0.0

        # Covariance with other bytes (track top correlated bytes)
        self.co_changes: Dict[int, int] = defaultdict(int)

    @property
    def change_frequency(self) -> float:
        return self.change_count / max(self.total_observations, 1)

    @property
    def n_unique_values(self) -> int:
        return len(self.values_seen)

    @property
    def is_binary(self) -> bool:
        return self.n_unique_values <= 3

    @property
    def category(self) -> str:
        freq = self.change_frequency
        if freq > 0.4:
            return "position"
        elif freq > 0.05:
            if self.increments > self.decrements * 3:
                return "counter_up"
            elif self.decrements > self.increments * 3:
                return "counter_down"
            return "dynamic"
        elif self.change_count > 0:
            if self.is_binary:
                return "state_flag"
            return "state"
        else:
            return "static"

    @property
    def semantic_label(self) -> str:
        """Best-guess semantic label based on observed behavior."""
        cat = self.category
        if cat == "position":
            if self.address % 2 == 0:
                return f"pos_x_{self.address:#04x}"
            return f"pos_y_{self.address:#04x}"
        elif cat == "counter_up":
            return f"score_or_timer_{self.address:#04x}"
        elif cat == "counter_down":
            return f"lives_or_health_{self.address:#04x}"
        elif cat == "state_flag":
            if self.changed_at_reward > 0:
                return f"item_collected_{self.address:#04x}"
            return f"flag_{self.address:#04x}"
        elif cat == "state":
            if self.changed_at_reward > 0:
                return f"progress_{self.address:#04x}"
            return f"state_{self.address:#04x}"
        return f"byte_{self.address:#04x}"


class RAMSemanticMapper:
    """
    Auto-discovers game objects from RAM observation patterns.

    Watches RAM changes and classifies bytes into semantic categories.
    Learns which bytes correlate with rewards, deaths, and each other.
    """

    def __init__(self, ram_size: int = 128):
        self._ram_size = ram_size
        self._profiles: Dict[int, RAMByteProfile] = {
            i: RAMByteProfile(i) for i in range(ram_size)
        }
        self._total_observations: int = 0
        self._prev_ram: Optional[np.ndarray] = None
        self._episode_start_ram: Optional[np.ndarray] = None

    def observe(
        self,
        ram: np.ndarray,
        action: int = 0,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        """
        Observe a RAM state and update byte profiles.

        Call this every frame during play (human or agent).
        """
        ram = np.asarray(ram, dtype=np.uint8).flatten()[:self._ram_size]
        self._total_observations += 1

        is_episode_start = (self._prev_ram is None)

        if self._prev_ram is not None:
            changed_indices = np.where(ram != self._prev_ram)[0]

            for idx in changed_indices:
                idx = int(idx)
                prof = self._profiles[idx]
                old_val = int(self._prev_ram[idx])
                new_val = int(ram[idx])

                prof.change_count += 1
                prof.values_seen.add(new_val)
                prof.min_value = min(prof.min_value, new_val)
                prof.max_value = max(prof.max_value, new_val)

                # Direction tracking
                if new_val > old_val:
                    prof.increments += 1
                elif new_val < old_val:
                    prof.decrements += 1
                if new_val == 0 and old_val != 0:
                    prof.resets += 1
                if (old_val == 0) != (new_val == 0):
                    prof.flips += 1

                # Event correlation
                if reward != 0:
                    prof.changed_at_reward += 1
                    prof.reward_total_when_changed += reward
                if done and reward <= 0:
                    prof.changed_at_death += 1

                # Co-change tracking
                for other_idx in changed_indices:
                    if other_idx != idx:
                        prof.co_changes[int(other_idx)] += 1

        # Track all bytes
        for i in range(len(ram)):
            self._profiles[i].total_observations += 1
            self._profiles[i].last_value = int(ram[i])
            self._profiles[i].values_seen.add(int(ram[i]))

        if is_episode_start:
            self._episode_start_ram = ram.copy()

        if done:
            # Track which bytes change between episode boundaries
            if self._episode_start_ram is not None:
                changed = np.where(ram != self._episode_start_ram)[0]
                for idx in changed:
                    self._profiles[int(idx)].changed_at_episode_start += 1
            self._prev_ram = None  # Reset for next episode
        else:
            self._prev_ram = ram.copy()

    def observe_batch(self, frames: List[Dict[str, Any]]) -> None:
        """Observe a batch of frames from a HumanRecorder recording."""
        for frame in frames:
            ram = np.array(frame.get("ram_snapshot", []), dtype=np.uint8)
            if len(ram) == 0:
                continue
            self.observe(
                ram,
                action=frame.get("action", 0),
                reward=frame.get("reward", 0.0),
                done=frame.get("done", False),
            )

    def get_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the semantic registry: categorized RAM bytes with labels.

        Returns:
            {
                "position": [{"addr": 0x31, "label": "pos_x_0x31", ...}, ...],
                "state_flag": [...],
                "counter_up": [...],
                ...
            }
        """
        registry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for addr, prof in sorted(self._profiles.items()):
            if prof.change_count == 0:
                continue  # Skip static bytes

            entry = {
                "addr": addr,
                "addr_hex": f"0x{addr:02X}",
                "label": prof.semantic_label,
                "category": prof.category,
                "change_freq": round(prof.change_frequency, 4),
                "n_values": prof.n_unique_values,
                "range": (prof.min_value, prof.max_value),
                "reward_correlated": prof.changed_at_reward > 0,
                "death_correlated": prof.changed_at_death > 0,
            }
            registry[prof.category].append(entry)

        return dict(registry)

    def get_subgoal_bytes(self) -> List[Dict[str, Any]]:
        """Get bytes that change when rewards happen — likely subgoal indicators."""
        subgoal = []
        for addr, prof in self._profiles.items():
            if prof.changed_at_reward > 0:
                subgoal.append({
                    "addr": addr,
                    "addr_hex": f"0x{addr:02X}",
                    "label": prof.semantic_label,
                    "reward_changes": prof.changed_at_reward,
                    "avg_reward": round(prof.reward_total_when_changed / prof.changed_at_reward, 2),
                    "is_binary": prof.is_binary,
                })
        return sorted(subgoal, key=lambda x: -x["reward_changes"])

    def get_position_pairs(self) -> List[Tuple[int, int]]:
        """
        Find likely (x, y) coordinate pairs among position bytes.

        Two position bytes that frequently co-change are likely x/y
        of the same entity.
        """
        position_bytes = [
            addr for addr, prof in self._profiles.items()
            if prof.category == "position"
        ]

        pairs = []
        for i, a in enumerate(position_bytes):
            for b in position_bytes[i + 1:]:
                co_count = self._profiles[a].co_changes.get(b, 0)
                if co_count > self._total_observations * 0.3:
                    pairs.append((a, b))
        return pairs

    def get_entity_groups(self) -> List[Dict[str, Any]]:
        """
        Group position bytes into entities (player, enemies, projectiles).

        Uses co-change frequency to cluster bytes that move together.
        """
        position_bytes = [
            addr for addr, prof in self._profiles.items()
            if prof.category == "position"
        ]

        if not position_bytes:
            return []

        # Simple clustering: bytes that co-change frequently are one entity
        assigned: Set[int] = set()
        entities: List[Dict[str, Any]] = []

        for addr in position_bytes:
            if addr in assigned:
                continue
            cluster = [addr]
            assigned.add(addr)

            for other in position_bytes:
                if other in assigned:
                    continue
                co_count = self._profiles[addr].co_changes.get(other, 0)
                if co_count > self._profiles[addr].change_count * 0.4:
                    cluster.append(other)
                    assigned.add(other)

            entity_type = "player" if entities == [] else f"entity_{len(entities)}"
            entities.append({
                "type": entity_type,
                "bytes": cluster,
                "labels": [self._profiles[b].semantic_label for b in cluster],
            })

        return entities

    def report(self) -> Dict[str, Any]:
        registry = self.get_registry()
        return {
            "total_observations": self._total_observations,
            "ram_size": self._ram_size,
            "active_bytes": sum(1 for p in self._profiles.values() if p.change_count > 0),
            "categories": {k: len(v) for k, v in registry.items()},
            "subgoal_bytes": len(self.get_subgoal_bytes()),
            "entity_groups": len(self.get_entity_groups()),
        }
