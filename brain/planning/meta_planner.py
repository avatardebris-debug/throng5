"""
meta_planner.py — Auto-select which cognitive system to use.

The agent has many systems:
  - Reactive (DQN/Striatum) — fast, habitual
  - Planning (SubgoalPlanner) — slow, deliberate
  - Rehearsal (RehearsalLoop) — offline practice
  - LLM Strategy — strategic consultation
  - Counterfactual — hindsight analysis

The MetaPlanner decides WHEN to use each, based on:
  - Current situation (familiar vs. novel)
  - Recent performance (improving vs. stuck)
  - Available resources (lives, time)
  - Task complexity (open exploration vs. precise puzzle)

Usage:
    meta = MetaPlanner(brain)
    mode = meta.decide()  # "reactive", "planning", "rehearse", "consult_llm"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class MetaPlanner:
    """
    Meta-controller that selects the appropriate cognitive system.

    Monitors the agent's performance and context to decide:
    - Use reactive mode in familiar territory
    - Switch to planning mode at bottlenecks
    - Enter rehearsal when stuck
    - Consult LLM when deeply stuck or in novel situations
    - Run counterfactuals after unexpected deaths
    """

    def __init__(self, brain=None):
        self.brain = brain

        # Performance tracking
        self._recent_rewards: List[float] = []
        self._recent_deaths: int = 0
        self._steps_since_progress: int = 0
        self._steps_since_death: int = 0
        self._current_mode: str = "reactive"

        # Thresholds
        self._stuck_threshold: int = 500    # Steps without progress → planning
        self._deep_stuck: int = 2000        # Steps without progress → rehearse
        self._llm_threshold: int = 5000     # Steps without progress → consult LLM
        self._death_burst: int = 3          # Deaths in 100 steps → planning

        # History
        self._mode_history: List[Dict[str, Any]] = []
        self._mode_durations: Dict[str, int] = defaultdict(int)
        self._transitions: int = 0
        self._step: int = 0

        # State novelty
        self._known_states: set = set()

    def observe(
        self,
        features: np.ndarray,
        reward: float,
        done: bool,
        action: int = 0,
    ) -> None:
        """Observe a step result and update internal tracking."""
        self._step += 1
        self._steps_since_progress += 1
        self._steps_since_death += 1

        # Track reward
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 100:
            self._recent_rewards.pop(0)

        # Progress detection
        if reward > 0:
            self._steps_since_progress = 0

        # Death detection
        if done and reward <= 0:
            self._recent_deaths += 1
            self._steps_since_death = 0

        # Decay death counter
        if self._step % 100 == 0:
            self._recent_deaths = max(0, self._recent_deaths - 1)

        # State novelty
        state_hash = hash(np.asarray(features).tobytes()) % 50000
        self._known_states.add(state_hash)

        self._mode_durations[self._current_mode] += 1

    def decide(self) -> str:
        """
        Decide which cognitive system to use.

        Returns one of:
          "reactive"    — normal DQN play (fast, habitual)
          "planning"    — use SubgoalPlanner (deliberate)
          "rehearse"    — enter rehearsal (practice bottleneck)
          "consult_llm" — ask Tetra for strategy
          "counterfactual" — analyze recent failures
          "explore"     — curiosity-driven exploration
        """
        old_mode = self._current_mode

        # Priority 1: Death burst → switch to planning
        if self._recent_deaths >= self._death_burst:
            self._current_mode = "counterfactual"

        # Priority 2: Very deeply stuck → consult LLM
        elif self._steps_since_progress >= self._llm_threshold:
            self._current_mode = "consult_llm"

        # Priority 3: Deeply stuck → rehearse
        elif self._steps_since_progress >= self._deep_stuck:
            self._current_mode = "rehearse"

        # Priority 4: Moderately stuck → switch to planning
        elif self._steps_since_progress >= self._stuck_threshold:
            self._current_mode = "planning"

        # Priority 5: Making progress → trust reactive
        elif self._steps_since_progress < 50:
            self._current_mode = "reactive"

        # Priority 6: Novel territory with no progress → explore
        elif len(self._known_states) < self._step * 0.1:
            # Very few unique states relative to steps → stuck in loop
            self._current_mode = "explore"

        # Default: stay in current mode
        if old_mode != self._current_mode:
            self._transitions += 1
            self._mode_history.append({
                "step": self._step,
                "from": old_mode,
                "to": self._current_mode,
                "reason": self._get_reason(),
            })

        return self._current_mode

    def _get_reason(self) -> str:
        if self._recent_deaths >= self._death_burst:
            return f"{self._recent_deaths} deaths in recent window"
        if self._steps_since_progress >= self._llm_threshold:
            return f"No progress for {self._steps_since_progress} steps"
        if self._steps_since_progress >= self._deep_stuck:
            return f"Stuck for {self._steps_since_progress} steps"
        if self._steps_since_progress >= self._stuck_threshold:
            return f"Stalling for {self._steps_since_progress} steps"
        if self._steps_since_progress < 50:
            return "Making progress"
        return "Default"

    def should_plan(self) -> bool:
        return self._current_mode in ("planning", "consult_llm")

    def should_rehearse(self) -> bool:
        return self._current_mode == "rehearse"

    def should_explore(self) -> bool:
        return self._current_mode == "explore"

    def should_consult_llm(self) -> bool:
        return self._current_mode == "consult_llm"

    def force_mode(self, mode: str) -> None:
        """Override the automatic decision."""
        old_mode = self._current_mode
        self._current_mode = mode
        if old_mode != mode:
            self._transitions += 1
            self._mode_history.append({
                "step": self._step,
                "from": old_mode,
                "to": mode,
                "reason": "forced",
            })

    def reset_progress(self) -> None:
        """Called when genuine progress is made."""
        self._steps_since_progress = 0
        self._recent_deaths = 0

    def report(self) -> Dict[str, Any]:
        avg_reward = (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards else 0.0
        )
        return {
            "current_mode": self._current_mode,
            "step": self._step,
            "steps_since_progress": self._steps_since_progress,
            "recent_deaths": self._recent_deaths,
            "avg_recent_reward": round(avg_reward, 4),
            "mode_durations": dict(self._mode_durations),
            "transitions": self._transitions,
            "known_states": len(self._known_states),
            "last_transitions": self._mode_history[-5:] if self._mode_history else [],
        }
