"""
Amygdala — Danger/loss detection from dream simulations.

Monitors DreamerEngine output for catastrophic futures.
Triggers emergency policy switches when all hypotheses predict loss.

Decision hierarchy:
  1. If ANY hypothesis shows positive future → no danger, let PolicyMonitor decide
  2. If ALL hypotheses show loss → danger signal, recommend override
  3. Override options: random exploration, archived policy, or LLM-generated policy

Feeds from PredictionErrorTracker (large errors = surprise → heightened alertness).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class DangerLevel(Enum):
    """Threat assessment levels."""
    SAFE = "safe"              # At least one hypothesis is positive
    CAUTION = "caution"        # Most hypotheses negative, some ok
    DANGER = "danger"          # All hypotheses negative
    CRITICAL = "critical"      # All hypotheses show catastrophic loss


class OverrideAction(Enum):
    """What to do when danger is detected."""
    NONE = "none"              # No override needed
    RANDOM_EXPLORE = "random"  # Switch to random exploration
    ARCHIVED_POLICY = "archive"  # Try a previously saved policy
    LLM_CONSTRUCT = "llm"     # Ask LLM to construct new policy


@dataclass
class DangerSignal:
    """Output from amygdala danger assessment."""
    level: DangerLevel
    confidence: float                    # 0-1, how sure we are
    recommended_action: OverrideAction
    reason: str
    all_hypotheses_negative: bool
    worst_predicted_reward: float
    best_predicted_reward: float
    n_catastrophic: int                  # Hypotheses with has_catastrophe

    @property
    def should_override(self) -> bool:
        """Should we force a policy switch?"""
        return self.level in (DangerLevel.DANGER, DangerLevel.CRITICAL)

    def summary(self) -> str:
        emoji = {
            DangerLevel.SAFE: "🟢",
            DangerLevel.CAUTION: "🟡",
            DangerLevel.DANGER: "🔴",
            DangerLevel.CRITICAL: "🚨",
        }[self.level]
        return (
            f"{emoji} {self.level.value} (conf={self.confidence:.2f}): "
            f"{self.reason} | "
            f"best={self.best_predicted_reward:+.2f}, "
            f"worst={self.worst_predicted_reward:+.2f}"
        )


class Amygdala:
    """
    Danger detection from dream simulations.

    Monitors DreamerEngine results and PredictionErrorTracker output
    to assess threat level and recommend overrides.

    Does NOT execute overrides — reports to MetaPolicyController which decides.
    """

    # Thresholds
    CATASTROPHE_REWARD = -1.0    # Below this = catastrophic step
    DANGER_THRESHOLD = -0.5      # All hypotheses below this = danger
    CAUTION_RATIO = 0.7          # >70% hypotheses negative = caution

    # Override cooldown (avoid thrashing)
    MIN_OVERRIDE_INTERVAL = 10   # Steps between overrides

    def __init__(self,
                 catastrophe_threshold: float = -1.0,
                 danger_threshold: float = -0.5,
                 surprise_weight: float = 0.3):
        """
        Args:
            catastrophe_threshold: Reward below this = catastrophic
            danger_threshold: All hypotheses below this = danger
            surprise_weight: How much prediction surprise affects alertness
        """
        self.catastrophe_threshold = catastrophe_threshold
        self.danger_threshold = danger_threshold
        self.surprise_weight = surprise_weight

        # State
        self._alertness = 0.0        # 0-1, amplified by prediction errors
        self._last_override_step = -999
        self._override_count = 0
        self._assessment_count = 0
        self._danger_history: List[DangerLevel] = []

    def assess_danger(self, dream_results: list,
                      surprise_level: float = 0.0,
                      current_step: int = 0) -> DangerSignal:
        """
        Assess danger from dream simulation results.

        Args:
            dream_results: List[DreamResult] from DreamerEngine.dream()
            surprise_level: 0-1 from PredictionErrorTracker
            current_step: Current step count (for cooldown)

        Returns:
            DangerSignal with threat assessment and recommendation
        """
        self._assessment_count += 1

        # Update alertness from prediction surprise
        self._alertness = (
            self._alertness * 0.9 +
            surprise_level * self.surprise_weight
        )

        if not dream_results:
            return self._safe_signal("No dream results to assess")

        # Analyze results
        total_rewards = [r.total_predicted_reward for r in dream_results]
        n_negative = sum(1 for r in total_rewards if r < 0)
        n_catastrophic = sum(1 for r in dream_results if r.has_catastrophe)
        all_negative = all(r < 0 for r in total_rewards)
        best_reward = max(total_rewards)
        worst_reward = min(total_rewards)

        # Average confidence from dream results
        avg_confidence = np.mean([r.confidence for r in dream_results])

        # Blend confidence with alertness
        effective_confidence = min(1.0, avg_confidence + self._alertness * 0.2)

        # Determine danger level
        if all_negative and worst_reward < self.catastrophe_threshold:
            level = DangerLevel.CRITICAL
            reason = (
                f"ALL {len(dream_results)} hypotheses predict loss, "
                f"worst={worst_reward:+.2f} (catastrophic)"
            )
        elif all_negative:
            level = DangerLevel.DANGER
            reason = (
                f"ALL {len(dream_results)} hypotheses predict loss "
                f"(best={best_reward:+.2f})"
            )
        elif n_negative / len(dream_results) >= self.CAUTION_RATIO:
            level = DangerLevel.CAUTION
            reason = (
                f"{n_negative}/{len(dream_results)} hypotheses negative"
            )
        else:
            level = DangerLevel.SAFE
            reason = (
                f"{len(dream_results) - n_negative}/"
                f"{len(dream_results)} hypotheses positive"
            )

        # Determine recommended action
        recommended = self._recommend_action(
            level, current_step, n_catastrophic
        )

        self._danger_history.append(level)
        # Keep bounded
        if len(self._danger_history) > 200:
            self._danger_history = self._danger_history[-200:]

        return DangerSignal(
            level=level,
            confidence=effective_confidence,
            recommended_action=recommended,
            reason=reason,
            all_hypotheses_negative=all_negative,
            worst_predicted_reward=worst_reward,
            best_predicted_reward=best_reward,
            n_catastrophic=n_catastrophic,
        )

    def should_override(self, danger: DangerSignal,
                        current_step: int = 0) -> bool:
        """
        Should we force a policy override?

        Considers danger level AND cooldown to prevent thrashing.
        """
        if not danger.should_override:
            return False

        # Cooldown check
        steps_since_override = current_step - self._last_override_step
        if steps_since_override < self.MIN_OVERRIDE_INTERVAL:
            return False

        # Critical always overrides (even during cooldown)
        if danger.level == DangerLevel.CRITICAL:
            return True

        # Danger overrides only if confident
        return danger.confidence >= 0.3

    def record_override(self, step: int):
        """Record that an override was executed at this step."""
        self._last_override_step = step
        self._override_count += 1

    def _recommend_action(self, level: DangerLevel,
                          current_step: int,
                          n_catastrophic: int) -> OverrideAction:
        """Choose what override action to recommend."""
        if level == DangerLevel.SAFE:
            return OverrideAction.NONE

        if level == DangerLevel.CAUTION:
            return OverrideAction.NONE  # Not severe enough

        # Danger or Critical
        if n_catastrophic > 0 and self._override_count >= 3:
            # We've tried overrides before and still failing
            # Escalate to LLM
            return OverrideAction.LLM_CONSTRUCT

        if self._override_count >= 2:
            # Random didn't help, try archived policy
            return OverrideAction.ARCHIVED_POLICY

        # First resort: random exploration
        return OverrideAction.RANDOM_EXPLORE

    def _safe_signal(self, reason: str) -> DangerSignal:
        return DangerSignal(
            level=DangerLevel.SAFE,
            confidence=0.0,
            recommended_action=OverrideAction.NONE,
            reason=reason,
            all_hypotheses_negative=False,
            worst_predicted_reward=0.0,
            best_predicted_reward=0.0,
            n_catastrophic=0,
        )

    @property
    def alertness(self) -> float:
        return self._alertness

    @property
    def recent_danger_ratio(self) -> float:
        """Fraction of recent assessments that were DANGER or CRITICAL."""
        if not self._danger_history:
            return 0.0
        recent = self._danger_history[-20:]
        n_dangerous = sum(
            1 for d in recent
            if d in (DangerLevel.DANGER, DangerLevel.CRITICAL)
        )
        return n_dangerous / len(recent)

    def reset(self):
        """Reset for new environment."""
        self._alertness = 0.0
        self._last_override_step = -999
        self._override_count = 0
        self._danger_history.clear()

    def summary(self) -> str:
        lines = [
            f"Amygdala:",
            f"  Assessments: {self._assessment_count}",
            f"  Overrides triggered: {self._override_count}",
            f"  Current alertness: {self._alertness:.2f}",
            f"  Recent danger ratio: {self.recent_danger_ratio:.2f}",
        ]
        return "\n".join(lines)
