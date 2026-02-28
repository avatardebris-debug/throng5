"""
safety.py — Hard safety constraints that survive transfer.

Unlike soft reward signals, safety constraints are RULES:
  - "Never walk into lava" (learned from death correlation)
  - "Don't push blocks off the edge" (irreversible)
  - "Avoid this region when enemy is patrolling" (temporal + spatial)

Constraints are learned from experience and can transfer between games
(e.g., "avoid falling" applies to both Montezuma and Lolo).

Usage:
    safety = SafetyConstraints()
    safety.learn_from_death(features, action, context="fell_into_pit")
    safe_actions = safety.filter_actions(features, [0, 1, 2, 3])
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Constraint:
    """A learned safety constraint."""
    name: str
    constraint_type: str  # "action_ban", "region_avoid", "sequence_ban"

    # When the constraint applies
    condition_features: Optional[np.ndarray] = None
    condition_ram_byte: Optional[int] = None
    condition_ram_value: Optional[int] = None

    # What's forbidden
    forbidden_actions: Set[int] = field(default_factory=set)
    forbidden_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    # Evidence
    death_count: int = 0
    violation_count: int = 0
    confidence: float = 0.0
    transferable: bool = False  # True if general enough to transfer

    @property
    def severity(self) -> float:
        """How dangerous is violating this constraint?"""
        return min(1.0, self.death_count / 5.0) * self.confidence


class SafetyConstraints:
    """
    Maintains and enforces learned safety rules.

    Rules are learned from:
    1. Death events (what action in what state caused death)
    2. Causal model (which actions are irreversible)
    3. Manual specification (game-specific rules)
    4. Transfer from other games (universal rules like "don't fall")
    """

    def __init__(self):
        self._constraints: Dict[str, Constraint] = {}
        self._death_history: List[Dict[str, Any]] = []

        # Quick lookup: state_hash → forbidden actions
        self._action_bans: Dict[int, Set[int]] = defaultdict(set)

        # Universal constraints (transfer between games)
        self._universal: List[Constraint] = []

        # Stats
        self._violations_prevented: int = 0
        self._total_checks: int = 0

    def learn_from_death(
        self,
        features: np.ndarray,
        action: int,
        context: str = "",
        ram: Optional[np.ndarray] = None,
    ) -> None:
        """
        Learn a constraint from a death event.

        If the same (state, action) causes death 3+ times,
        create a hard ban.
        """
        state_hash = hash(np.asarray(features).tobytes()) % 100000
        death_key = f"death_{state_hash}_{action}"

        self._death_history.append({
            "state_hash": state_hash,
            "action": action,
            "context": context,
        })

        if death_key in self._constraints:
            self._constraints[death_key].death_count += 1
            self._constraints[death_key].confidence = min(
                1.0, self._constraints[death_key].death_count / 5.0
            )
        else:
            self._constraints[death_key] = Constraint(
                name=death_key,
                constraint_type="action_ban",
                forbidden_actions={action},
                death_count=1,
                confidence=0.2,
            )

        # After 3 deaths, create a hard ban
        if self._constraints[death_key].death_count >= 3:
            self._action_bans[state_hash].add(action)

            # Check if this is a transferable rule
            if context in ("fell", "pit", "lava", "spike", "void"):
                self._constraints[death_key].transferable = True

    def learn_from_causal(
        self,
        action: int,
        is_irreversible: bool,
        is_dangerous: bool,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Learn constraints from the CausalModel."""
        if is_irreversible and is_dangerous:
            state_hash = 0
            if features is not None:
                state_hash = hash(np.asarray(features).tobytes()) % 100000
            key = f"causal_{state_hash}_{action}"
            self._constraints[key] = Constraint(
                name=key,
                constraint_type="action_ban",
                forbidden_actions={action},
                confidence=0.7,
            )
            self._action_bans[state_hash].add(action)

    def add_region_constraint(
        self,
        name: str,
        x1: int, y1: int, x2: int, y2: int,
        forbidden_actions: Optional[Set[int]] = None,
    ) -> None:
        """Add a spatial region constraint."""
        self._constraints[name] = Constraint(
            name=name,
            constraint_type="region_avoid",
            forbidden_region=(x1, y1, x2, y2),
            forbidden_actions=forbidden_actions or set(),
            confidence=1.0,
        )

    def filter_actions(
        self,
        features: np.ndarray,
        available_actions: List[int],
        player_pos: Optional[Tuple[int, int]] = None,
    ) -> List[int]:
        """
        Remove unsafe actions from the available set.

        Always returns at least one action (the least-dangerous one).
        """
        self._total_checks += 1
        state_hash = hash(np.asarray(features).tobytes()) % 100000

        forbidden = set()

        # Check state-specific bans
        if state_hash in self._action_bans:
            forbidden.update(self._action_bans[state_hash])

        # Check region constraints
        if player_pos is not None:
            px, py = player_pos
            for c in self._constraints.values():
                if c.constraint_type == "region_avoid" and c.forbidden_region:
                    x1, y1, x2, y2 = c.forbidden_region
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        forbidden.update(c.forbidden_actions)

        safe = [a for a in available_actions if a not in forbidden]

        if not safe:
            # All actions forbidden — return the least-dangerous one
            safe = available_actions

        if len(safe) < len(available_actions):
            self._violations_prevented += 1

        return safe

    def is_safe(self, features: np.ndarray, action: int) -> bool:
        """Quick check if an action is safe in this state."""
        state_hash = hash(np.asarray(features).tobytes()) % 100000
        return action not in self._action_bans.get(state_hash, set())

    def get_transferable(self) -> List[Constraint]:
        """Get constraints that can transfer to other games."""
        return [c for c in self._constraints.values() if c.transferable]

    def report(self) -> Dict[str, Any]:
        return {
            "total_constraints": len(self._constraints),
            "action_bans": sum(len(v) for v in self._action_bans.values()),
            "violations_prevented": self._violations_prevented,
            "total_checks": self._total_checks,
            "transferable": len(self.get_transferable()),
            "total_deaths_learned": len(self._death_history),
        }
