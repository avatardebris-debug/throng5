"""
procedural_memory.py — Motor skills separate from declarative plans.

Declarative: "I know I need to get the key" (planning layer)
Procedural: "I know HOW to jump across a 3-tile gap" (this module)

Procedural memories are learned motor programs — sequences of actions
that achieve specific physical outcomes. Unlike action chains (which are
state-specific), procedural skills are transferable:
  - "Jump right" works on any platform
  - "Dodge left" works against any enemy
  - "Climb ladder" works on any ladder

Stored as action-outcome pairs with reliability tracking.

Usage:
    memory = ProceduralMemory()
    memory.record(actions=[2, 0, 0], outcome="moved_right_3", success=True)
    skill = memory.recall("jump_right")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MotorProgram:
    """A learned motor skill — actions that produce a specific outcome."""
    name: str
    actions: List[int]
    outcome_type: str = ""  # "move_right", "jump_gap", "climb", "dodge"

    # Context: when does this skill apply?
    context_features: Optional[np.ndarray] = None  # State features when learned
    requires_ground: bool = False
    requires_proximity: bool = False

    # Performance
    executions: int = 0
    successes: int = 0
    avg_reward: float = 0.0
    last_used_step: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.executions, 1)

    @property
    def reliability(self) -> float:
        """Bayesian credibility: (successes + 1) / (total + 2)"""
        return (self.successes + 1) / (self.executions + 2)


class ProceduralMemory:
    """
    Library of learned motor programs.

    Unlike ActionChainStore (state-hash → specific chain), ProceduralMemory
    stores reusable motor patterns indexed by outcome type:
      "jump_right" → [jump, right, right]
      "dodge_left" → [left, left, left]
      "climb_up"   → [up, up, up, up]

    Motor programs are discovered from:
    1. Observing successful action sequences
    2. Extracting common sub-patterns across different states
    3. Human demonstration analysis
    """

    def __init__(self, max_programs: int = 500):
        self._programs: Dict[str, MotorProgram] = {}
        self._max = max_programs

        # Pattern discovery
        self._action_buffer: List[int] = []
        self._buffer_states: List[np.ndarray] = []
        self._buffer_rewards: List[float] = []
        self._common_patterns: Dict[tuple, int] = defaultdict(int)

        # Stats
        self._total_recordings: int = 0
        self._patterns_discovered: int = 0

    def record(
        self,
        actions: List[int],
        outcome: str,
        success: bool,
        reward: float = 0.0,
        context_features: Optional[np.ndarray] = None,
        step: int = 0,
    ) -> str:
        """
        Record a motor program from observed actions.

        If a program with this outcome already exists, update its stats.
        If the new version has higher success rate, replace the actions.
        """
        self._total_recordings += 1

        if outcome in self._programs:
            prog = self._programs[outcome]
            prog.executions += 1
            if success:
                prog.successes += 1
            prog.avg_reward = (
                prog.avg_reward * (prog.executions - 1) + reward
            ) / prog.executions
            prog.last_used_step = step

            # Replace if new sequence is shorter and still succeeded
            if success and len(actions) < len(prog.actions):
                prog.actions = list(actions)
        else:
            prog = MotorProgram(
                name=outcome,
                actions=list(actions),
                outcome_type=self._classify_outcome(actions),
                context_features=context_features.copy() if context_features is not None else None,
                executions=1,
                successes=1 if success else 0,
                avg_reward=reward,
                last_used_step=step,
            )
            self._programs[outcome] = prog

        return outcome

    def recall(self, outcome: str) -> Optional[MotorProgram]:
        """Recall a motor program by outcome name."""
        return self._programs.get(outcome)

    def recall_by_type(self, outcome_type: str) -> List[MotorProgram]:
        """Recall all programs of a given type."""
        return [
            p for p in self._programs.values()
            if p.outcome_type == outcome_type
        ]

    def get_best(self, outcome_type: str) -> Optional[MotorProgram]:
        """Get the most reliable program of a given type."""
        candidates = self.recall_by_type(outcome_type)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.reliability)

    def observe_transition(
        self,
        action: int,
        features_before: np.ndarray,
        features_after: np.ndarray,
        reward: float,
    ) -> None:
        """
        Buffer transitions for pattern discovery.

        After enough observations, automatically extracts common
        action sub-sequences.
        """
        self._action_buffer.append(action)
        self._buffer_states.append(features_before)
        self._buffer_rewards.append(reward)

        # Extract patterns every 500 steps
        if len(self._action_buffer) >= 500:
            self._discover_patterns()
            self._action_buffer = self._action_buffer[-100:]
            self._buffer_states = self._buffer_states[-100:]
            self._buffer_rewards = self._buffer_rewards[-100:]

    def _discover_patterns(self) -> None:
        """
        Find recurring action sub-sequences that correlate with reward.

        Uses n-gram analysis of the action buffer and correlates
        with reward spikes.
        """
        actions = self._action_buffer
        rewards = self._buffer_rewards

        # Find 3-to-8 length patterns
        for length in range(3, 9):
            for i in range(len(actions) - length):
                pattern = tuple(actions[i:i + length])
                segment_reward = sum(rewards[i:i + length])

                if segment_reward > 0:
                    self._common_patterns[pattern] += 1

        # Promote frequent positive-reward patterns to motor programs
        for pattern, count in self._common_patterns.items():
            if count >= 3:  # Seen at least 3 times with positive reward
                name = f"pattern_{hash(pattern) % 10000}"
                if name not in self._programs:
                    self._programs[name] = MotorProgram(
                        name=name,
                        actions=list(pattern),
                        outcome_type=self._classify_outcome(list(pattern)),
                        executions=count,
                        successes=count,
                    )
                    self._patterns_discovered += 1

        self._common_patterns.clear()

    def _classify_outcome(self, actions: List[int]) -> str:
        """Classify a sequence of actions by its dominant direction/type."""
        if not actions:
            return "unknown"
        counts = defaultdict(int)
        for a in actions:
            counts[a] += 1
        dominant = max(counts, key=counts.get)
        mapping = {0: "move_right", 1: "move_left", 2: "move_up", 3: "move_down"}
        return mapping.get(dominant, f"action_{dominant}")

    def export_to_skill_library(self, skill_library) -> int:
        """Export reliable motor programs as skills in the SkillLibrary."""
        exported = 0
        for prog in self._programs.values():
            if prog.reliability > 0.5 and prog.executions >= 3:
                exported += 1
        return exported

    def report(self) -> Dict[str, Any]:
        reliable = sum(1 for p in self._programs.values() if p.reliability > 0.7)
        return {
            "total_programs": len(self._programs),
            "reliable_programs": reliable,
            "total_recordings": self._total_recordings,
            "patterns_discovered": self._patterns_discovered,
            "buffer_size": len(self._action_buffer),
        }
