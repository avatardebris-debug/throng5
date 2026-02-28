"""
self_model.py — Agent's model of its own capabilities.

The agent needs to know what it CAN do to plan effectively:
  - "Can I jump across a 5-tile gap?" → check procedural memory
  - "Can I survive this enemy?" → check historical success rate
  - "Can I reach that position from here?" → check landmark graph

This avoids planning impossible routes and enables realistic
confidence estimation.

Usage:
    model = SelfModel(brain)
    can_do = model.can_reach(target_features)
    capabilities = model.get_capabilities()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np


class SelfModel:
    """
    The agent's model of its own capabilities and limitations.

    Tracks:
    - What motor skills the agent has mastered
    - Which game areas are reachable vs. unreachable
    - Historical success rates for different action types
    - Current resource state (lives, items, power-ups)

    Used by the SubgoalPlanner to produce feasible plans.
    """

    def __init__(self, brain=None):
        self.brain = brain

        # Capability tracking
        self._action_success: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )
        self._reachable_landmarks: Set[int] = set()
        self._unreachable_landmarks: Set[int] = set()

        # Resource tracking
        self._current_lives: int = 3
        self._has_items: Dict[str, bool] = {}
        self._has_power: Dict[str, bool] = {}

        # Physical limits (learned from experience)
        self._max_jump_distance: float = 0.0
        self._max_fall_survive: float = 0.0
        self._move_speed: float = 1.0

        # Historical
        self._total_deaths: int = 0
        self._death_causes: Dict[str, int] = defaultdict(int)
        self._farthest_x: float = 0.0
        self._farthest_y: float = 0.0
        self._areas_visited: Set[int] = set()

    def record_action_result(
        self,
        action: int,
        success: bool,
        context: str = "",
    ) -> None:
        """Record whether an action succeeded in a given context."""
        key = f"{action}_{context}" if context else str(action)
        self._action_success[action]["attempts"] += 1
        if success:
            self._action_success[action]["successes"] += 1

    def record_death(self, cause: str = "unknown", features: Optional[np.ndarray] = None) -> None:
        self._total_deaths += 1
        self._death_causes[cause] += 1

    def record_position(self, x: float, y: float, area_hash: int = 0) -> None:
        self._farthest_x = max(self._farthest_x, x)
        self._farthest_y = max(self._farthest_y, y)
        if area_hash:
            self._areas_visited.add(area_hash)

    def can_perform(self, action: int) -> float:
        """Probability that this action will succeed (0-1)."""
        stats = self._action_success.get(action, {"attempts": 0, "successes": 0})
        if stats["attempts"] == 0:
            return 0.5  # Unknown — assume 50/50
        return stats["successes"] / stats["attempts"]

    def can_reach(self, landmark_hash: int) -> Optional[bool]:
        """
        Whether the agent believes it can reach a landmark.
        True = proven reachable, False = proven unreachable, None = unknown.
        """
        if landmark_hash in self._reachable_landmarks:
            return True
        if landmark_hash in self._unreachable_landmarks:
            return False
        return None

    def mark_reachable(self, landmark_hash: int) -> None:
        self._reachable_landmarks.add(landmark_hash)
        self._unreachable_landmarks.discard(landmark_hash)

    def mark_unreachable(self, landmark_hash: int) -> None:
        self._unreachable_landmarks.add(landmark_hash)

    def update_resources(self, lives: int = -1, items: Optional[Dict[str, bool]] = None) -> None:
        if lives >= 0:
            self._current_lives = lives
        if items:
            self._has_items.update(items)

    def get_capabilities(self) -> Dict[str, Any]:
        """Summary of what the agent can do."""
        reliable_actions = [
            a for a, stats in self._action_success.items()
            if stats["attempts"] >= 5 and stats["successes"] / stats["attempts"] > 0.6
        ]
        return {
            "reliable_actions": reliable_actions,
            "reachable_landmarks": len(self._reachable_landmarks),
            "unreachable_landmarks": len(self._unreachable_landmarks),
            "areas_visited": len(self._areas_visited),
            "max_jump": self._max_jump_distance,
            "farthest_x": self._farthest_x,
            "farthest_y": self._farthest_y,
            "lives": self._current_lives,
            "items": dict(self._has_items),
        }

    def should_attempt(self, plan_step: Dict[str, Any]) -> float:
        """
        Estimate probability of success for a planned action.

        Returns 0-1 confidence that the agent can execute this step.
        """
        confidence = 0.5  # Base unknown

        # Boost if landmark is known reachable
        target = plan_step.get("to", 0)
        if target in self._reachable_landmarks:
            confidence += 0.3

        # Reduce if landmark is known unreachable
        if target in self._unreachable_landmarks:
            confidence -= 0.4

        # Factor in plan step confidence
        plan_conf = plan_step.get("confidence", 0.5)
        confidence = (confidence + plan_conf) / 2

        return max(0.0, min(1.0, confidence))

    def report(self) -> Dict[str, Any]:
        return {
            **self.get_capabilities(),
            "total_deaths": self._total_deaths,
            "top_death_causes": dict(sorted(
                self._death_causes.items(), key=lambda x: -x[1],
            )[:5]),
        }
