"""
mode_controller.py — Hysteresis-based mode switching driven by ThreatEstimator.

Translates a continuous threat signal into a discrete operating mode:
  EXPLORE  → high entropy, hypothesis-testing, ε=0.35
  EXECUTE  → current hypothesis, normal ε=0.15
  SURVIVE  → conservative, flat-board priority, ε=0.05

Hysteresis prevents flickering: once SURVIVE triggers, it stays until
threat drops below the EXIT threshold for N consecutive steps.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class ModeController:
    """
    Smooth, hysteresis-controlled mode switcher.

    Modes: EXPLORE → EXECUTE → SURVIVE
    """

    MODES = ('EXPLORE', 'EXECUTE', 'SURVIVE')

    # Epsilon (exploration rate) per mode
    EPSILON = {'EXPLORE': 0.35, 'EXECUTE': 0.15, 'SURVIVE': 0.05}

    # Reward shaping multiplier per mode (applied to episode reward)
    REWARD_SCALE = {'EXPLORE': 1.0, 'EXECUTE': 1.0, 'SURVIVE': 0.8}

    def __init__(self,
                 enter_survive: float = 0.60,
                 exit_survive: float  = 0.35,
                 enter_explore: float = 0.20,
                 hysteresis_steps: int = 5):
        """
        Args:
            enter_survive:    threat >= this → switch to SURVIVE
            exit_survive:     threat < this for hysteresis_steps → exit SURVIVE
            enter_explore:    threat < this → switch to EXPLORE
            hysteresis_steps: Steps below exit_survive needed to leave SURVIVE
        """
        self.enter_survive   = enter_survive
        self.exit_survive    = exit_survive
        self.enter_explore   = enter_explore
        self.hysteresis_steps = hysteresis_steps

        self._mode           = 'EXECUTE'
        self._survive_count  = 0   # steps spent in SURVIVE this activation
        self._below_exit     = 0   # consecutive steps below exit threshold

        # History for logging
        self.mode_counts   = {'EXPLORE': 0, 'EXECUTE': 0, 'SURVIVE': 0}
        self.transitions   = []    # [(step, from_mode, to_mode, threat)]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def epsilon(self) -> float:
        return self.EPSILON[self._mode]

    def update(self, threat: float, step: int = 0) -> str:
        """
        Update mode based on new threat level.

        Args:
            threat: float in [0, 1] from ThreatEstimator.predict()
            step:   current step number (for logging)

        Returns:
            Current mode string after update.
        """
        prev = self._mode

        if self._mode == 'SURVIVE':
            self._survive_count += 1
            if threat < self.exit_survive:
                self._below_exit += 1
            else:
                self._below_exit = 0

            if self._below_exit >= self.hysteresis_steps:
                # Enough calm steps — exit SURVIVE
                self._mode       = 'EXECUTE'
                self._below_exit = 0
                self._survive_count = 0
        else:
            # Normal mode logic
            if threat >= self.enter_survive:
                self._mode = 'SURVIVE'
                self._below_exit = 0
                self._survive_count = 0
            elif threat < self.enter_explore:
                self._mode = 'EXPLORE'
            else:
                self._mode = 'EXECUTE'

        self.mode_counts[self._mode] += 1
        if prev != self._mode:
            self.transitions.append((step, prev, self._mode, round(threat, 3)))

        return self._mode

    def reset_episode(self):
        """Call at the start of each episode to reset per-episode state."""
        self._mode          = 'EXECUTE'
        self._survive_count = 0
        self._below_exit    = 0

    def mode_summary(self) -> dict:
        """Return mode distribution over all calls."""
        total = sum(self.mode_counts.values()) or 1
        return {m: round(c / total, 3) for m, c in self.mode_counts.items()}

    def __repr__(self):
        return (f"ModeController(mode={self._mode}, "
                f"survive_count={self._survive_count}, "
                f"below_exit={self._below_exit})")
