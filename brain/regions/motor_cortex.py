"""
motor_cortex.py — Fast Action Execution & Heuristic Fallback Region.

Responsible for:
  - Executing the final action selected by Striatum
  - Providing heuristic fallback actions when Striatum is halted
  - Running within the fast-path time budget (< 16.7ms for 60fps)
  - Compressing learned strategies into fast lookup tables

When the Amygdala halts higher functions, Motor Cortex continues running
using pre-compiled heuristics from previous dream/overnight sessions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion


class MotorCortex(BrainRegion):
    """
    Fast action execution with heuristic fallback.

    Always runs on the fast path. Never halted by amygdala.
    If the Striatum or higher regions are halted, Motor Cortex
    falls back to heuristic actions.
    """

    def __init__(self, bus: MessageBus, n_actions: int = 18):
        super().__init__("motor_cortex", bus)
        self.n_actions = n_actions

        # Heuristic table: state-hash → preferred action
        self._heuristics: Dict[int, int] = {}

        # Last action from Striatum
        self._last_striatum_action: Optional[int] = None
        self._fallback_count = 0
        self._total_actions = 0

    def _on_emergency(self, message) -> None:
        """Motor Cortex NEVER halts — override the default behavior."""
        pass  # Ignore halt signals; we're the last line of defense

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the final action.

        Expected inputs:
            striatum_action: Optional[int] — action from Striatum
            features: Optional[np.ndarray] — state features (for heuristic lookup)
            striatum_halted: bool — True if Striatum is currently halted
        """
        striatum_action = inputs.get("striatum_action")
        features = inputs.get("features")
        striatum_halted = inputs.get("striatum_halted", False)

        self._total_actions += 1

        if striatum_action is not None and not striatum_halted:
            # Normal path: use Striatum's action
            self._last_striatum_action = striatum_action
            action = striatum_action
            source = "striatum"
        else:
            # Fallback: use heuristic
            action = self._heuristic_action(features)
            source = "heuristic"
            self._fallback_count += 1

        return {
            "action": action,
            "source": source,
            "fallback_ratio": self._fallback_count / max(1, self._total_actions),
        }

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Learn heuristics from successful action sequences.

        Expected experience:
            features: np.ndarray — state that led to success
            action: int — action that was taken
            reward: float — reward received
        """
        features = experience.get("features")
        action = experience.get("action")
        reward = experience.get("reward", 0.0)

        if features is not None and reward > 0:
            # Store as heuristic: hash features → action
            key = hash(features.tobytes()) if hasattr(features, "tobytes") else hash(str(features))
            self._heuristics[key] = action

            # Limit heuristic table size
            if len(self._heuristics) > 10000:
                # Remove oldest entries (dict preserves insertion order in Python 3.7+)
                keys = list(self._heuristics.keys())
                for k in keys[:1000]:
                    del self._heuristics[k]

        return {"heuristic_count": len(self._heuristics)}

    def _heuristic_action(self, features: Optional[np.ndarray]) -> int:
        """Look up heuristic action or fall back to random."""
        if features is not None:
            key = hash(features.tobytes()) if hasattr(features, "tobytes") else hash(str(features))
            if key in self._heuristics:
                return self._heuristics[key]

        # Last resort: repeat last known good action or random
        if self._last_striatum_action is not None:
            return self._last_striatum_action
        return int(np.random.randint(self.n_actions))

    def install_heuristics(self, heuristics: Dict[int, int]) -> None:
        """Load pre-compiled heuristics from overnight/dream session."""
        self._heuristics.update(heuristics)

    def report(self) -> Dict[str, Any]:
        base = super().report()
        return {
            **base,
            "heuristic_count": len(self._heuristics),
            "fallback_ratio": round(self._fallback_count / max(1, self._total_actions), 3),
            "total_actions": self._total_actions,
        }
