"""
execution_profiler.py — Sub-ganglia for execution-level learning.

Philosophy
----------
This is NOT a lookup table. It is a soft nudge signal that:

  1. Learns gradually via EMA (not hard statistics)
     — methods improve slowly, no abrupt switches
  2. Gates on context-match
     — nudge strength ∝ how similar current state is to past successes
     — novel context → nudge = 0, defer to main policy
  3. Detects stochasticity
     — if the same method produces wildly different outcomes, flag it
     — high variance → nudge = 0, don't pretend you know
  4. Never replaces the main policy
     — nudge cap = 0.15 (half the dreamer's 0.3)
     — the main policy can always override

Failure attribution
-------------------
Each attempt is tagged with a failure_mode from:
  "timing"      — right action, wrong moment
  "sequence"    — wrong order of sub-actions
  "prediction"  — world model was wrong about outcome
  "stochastic"  — same inputs, different outcomes (RNG, enemy AI)
  "mechanical"  — input lag, frame drop (detected via timing jitter)
  "novel"       — context too far from training distribution
  "success"     — it worked

The failure_mode is inferred heuristically (not ground truth), and is
reported in ExecutionNudge.reason so the caller can log it.

Robotics / real-world safety
-----------------------------
The novelty fallback is the key safety property. When context_match < 0.3,
the profiler returns nudge_strength=0.0 and reason="novel_context". The
main policy runs unassisted. This prevents the profiler from confidently
applying a learned bias in a situation it has never seen — which is exactly
the failure mode that would hurt a real robot.

Usage
-----
    profiler = ExecutionProfiler()

    # After each step:
    profiler.record_attempt(
        goal="clear_line",
        method="stack_flat",
        state=compressed_state,
        success=reward > 0,
        outcome_value=reward,
    )

    # Before action selection:
    nudge = profiler.get_execution_nudge(
        goal="clear_line",
        state=compressed_state,
        candidate_methods=["stack_flat", "build_tower", "fill_gaps"],
    )
    if nudge.strength > 0:
        # Apply nudge.strength as a small bias toward nudge.method
        scores[nudge.method_index] += nudge.strength
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from throng4.basal_ganglia.mini_snn import MiniSNN


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionNudge:
    """
    Soft execution nudge from the ExecutionProfiler.

    strength = 0.0 means "defer to main policy entirely".
    strength > 0 means "lean slightly toward method_index".
    strength is always in [0.0, MAX_NUDGE_STRENGTH].
    """
    method: str               # Name of the recommended method
    method_index: int         # Index into candidate_methods list
    strength: float           # 0.0 = no nudge, 0.15 = max nudge
    reason: str               # Why this strength: "learned" | "novel_context" |
                              #   "stochastic" | "insufficient_data" | "no_methods"
    context_match: float      # 0.0–1.0: how familiar is the current state
    stochastic_flag: bool     # True if this goal/method is highly stochastic


@dataclass
class MethodStats:
    """
    Per-(goal, method) execution statistics.

    All learning is via EMA to prevent brittle over-fitting to past data.
    Context matching is done via a shared per-goal SNN (not per-method),
    stored in ExecutionProfiler._goal_snns.
    """
    # EMA success rate (α = EMA_ALPHA)
    ema_success: float = 0.5          # Start neutral, not optimistic
    # EMA of outcome values (reward magnitude)
    ema_outcome: float = 0.0
    # EMA of outcome variance (for stochasticity detection)
    ema_variance: float = 0.0
    # EMA of prediction error magnitude (for attribution)
    ema_pred_error: float = 0.0
    # Total attempts (for data sufficiency check)
    n_attempts: int = 0
    # Recent outcomes for variance estimation (capped deque)
    recent_outcomes: deque = field(
        default_factory=lambda: deque(maxlen=20)
    )
    # Failure mode counts
    failure_modes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_NUDGE_STRENGTH = 0.15      # Hard cap — never override main policy
EMA_ALPHA = 0.1                # Learning rate for EMA updates (slow, stable)
MIN_ATTEMPTS = 5               # Minimum attempts before nudge is trusted
NOVELTY_THRESHOLD = 0.3        # context_match below this → nudge = 0
STOCHASTIC_THRESHOLD = 0.25    # outcome variance above this → nudge = 0


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionProfiler
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionProfiler:
    """
    Sub-ganglia for execution-level learning.

    Tracks which execution methods work for which goals, conditioned on
    the current state context. Provides soft nudges toward better methods
    while gracefully deferring to the main policy under novelty or
    stochasticity.

    See module docstring for full design rationale.
    """

    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        # stats[(goal, method)] → MethodStats
        self._stats: Dict[Tuple[str, str], MethodStats] = defaultdict(MethodStats)
        # One shared SNN per goal (not per method) — context match is goal-level
        # This means get_execution_nudge calls the SNN exactly once per query
        self._goal_snns: Dict[str, MiniSNN] = {}
        # Total nudges issued and deferred
        self._nudges_issued: int = 0
        self._nudges_deferred: int = 0
        self._novelty_deferrals: int = 0
        self._stochastic_deferrals: int = 0
        # Log of recent nudges for debugging
        self._nudge_log: deque = deque(maxlen=100)

    # ── Recording ────────────────────────────────────────────────────────────

    def record_attempt(
        self,
        goal: str,
        method: str,
        state: np.ndarray,
        success: bool,
        outcome_value: float = 0.0,
        failure_mode: Optional[str] = None,
        predicted_reward: Optional[float] = None,
    ) -> None:
        """
        Record the outcome of one execution attempt.

        Args:
            goal:             What the agent was trying to do ("clear_line").
            method:           How it tried to do it ("stack_flat").
            state:            Compressed state at the time of the attempt.
            success:          Did the attempt achieve the goal?
            outcome_value:    Reward or score delta from this attempt.
            failure_mode:     Optional pre-labelled failure mode. If None,
                              the profiler infers it from prediction_error.
            predicted_reward: Dreamer's predicted reward for this action.
                              Used for prediction-error-based attribution:
                              large negative error → prediction failure
                              near-zero error      → execution failure
        """
        key = (goal, method)
        s = self._stats[key]

        # Lazy-create goal-level SNN (shared across all methods for this goal)
        if goal not in self._goal_snns:
            self._goal_snns[goal] = MiniSNN(input_dim=self.state_dim, n_neurons=400)

        # EMA updates
        outcome_binary = 1.0 if success else 0.0
        s.ema_success += EMA_ALPHA * (outcome_binary - s.ema_success)
        s.ema_outcome += EMA_ALPHA * (outcome_value - s.ema_outcome)

        # Variance update (EMA of squared deviation from EMA outcome)
        deviation_sq = (outcome_value - s.ema_outcome) ** 2
        s.ema_variance += EMA_ALPHA * (deviation_sq - s.ema_variance)

        # Prediction error tracking
        if predicted_reward is not None:
            pred_error = outcome_value - predicted_reward
            s.ema_pred_error += EMA_ALPHA * (abs(pred_error) - s.ema_pred_error)
        else:
            pred_error = None

        # SNN update — reward as dopamine signal (goal-level SNN, shared)
        state_arr = self._ensure_state(state)
        self._goal_snns[goal].process(state_arr, reward=outcome_value)

        # Attempt count and recent outcomes
        s.n_attempts += 1
        s.recent_outcomes.append(outcome_value)

        # Failure mode tracking
        mode = failure_mode or self._infer_failure_mode(
            success, outcome_value, pred_error, s
        )
        s.failure_modes[mode] += 1

    # ── Nudge ─────────────────────────────────────────────────────────────────

    def get_execution_nudge(
        self,
        goal: str,
        state: np.ndarray,
        candidate_methods: List[str],
    ) -> ExecutionNudge:
        """
        Get a soft nudge toward the best execution method for this goal.

        Returns nudge.strength = 0.0 (defer to main policy) when:
          - context_match < NOVELTY_THRESHOLD (novel situation)
          - outcome variance > STOCHASTIC_THRESHOLD (inherently noisy)
          - fewer than MIN_ATTEMPTS recorded for any method
          - no candidate methods provided

        Otherwise returns a small bias (0.0–0.15) toward the best method.

        Args:
            goal:              What the agent is trying to do.
            state:             Current compressed state.
            candidate_methods: Methods to choose among.

        Returns:
            ExecutionNudge with strength, method, reason, and diagnostics.
        """
        if not candidate_methods:
            return ExecutionNudge(
                method="", method_index=0, strength=0.0,
                reason="no_methods", context_match=0.0, stochastic_flag=False,
            )

        state_arr = self._ensure_state(state)

        # ── Single SNN call for this goal (shared context match) ──────────────
        # The SNN is goal-level: one call tells us how familiar this state is
        # for this goal, regardless of which method we're evaluating.
        goal_snn = self._goal_snns.get(goal)
        shared_match = goal_snn.process(state_arr, reward=0.0) if goal_snn else 0.5

        # Score each candidate method using the shared context match
        best_method = candidate_methods[0]
        best_idx = 0
        best_score = -np.inf
        best_match = shared_match
        best_variance = 0.0
        has_data = False

        for i, method in enumerate(candidate_methods):
            key = (goal, method)
            if key not in self._stats:
                continue
            s = self._stats[key]
            if s.n_attempts < MIN_ATTEMPTS:
                continue

            has_data = True
            # Score = EMA success weighted by shared context match
            score = s.ema_success * shared_match

            if score > best_score:
                best_score = score
                best_method = method
                best_idx = i
                best_variance = s.ema_variance

        # ── Gate 1: Insufficient data ─────────────────────────────────────
        if not has_data:
            self._nudges_deferred += 1
            return ExecutionNudge(
                method=candidate_methods[0], method_index=0, strength=0.0,
                reason="insufficient_data", context_match=0.0,
                stochastic_flag=False,
            )

        # ── Gate 2: Novelty — context too unfamiliar ──────────────────────
        if best_match < NOVELTY_THRESHOLD:
            self._nudges_deferred += 1
            self._novelty_deferrals += 1
            return ExecutionNudge(
                method=best_method, method_index=best_idx, strength=0.0,
                reason="novel_context", context_match=best_match,
                stochastic_flag=False,
            )

        # ── Gate 3: Stochasticity — outcome too unpredictable ─────────────
        stochastic = best_variance > STOCHASTIC_THRESHOLD
        if stochastic:
            self._nudges_deferred += 1
            self._stochastic_deferrals += 1
            return ExecutionNudge(
                method=best_method, method_index=best_idx, strength=0.0,
                reason="stochastic", context_match=best_match,
                stochastic_flag=True,
            )

        # ── Compute nudge strength ─────────────────────────────────────────
        # Strength scales with context_match and EMA success,
        # capped at MAX_NUDGE_STRENGTH.
        raw_strength = best_match * best_score * MAX_NUDGE_STRENGTH
        strength = float(np.clip(raw_strength, 0.0, MAX_NUDGE_STRENGTH))

        self._nudges_issued += 1
        self._nudge_log.append({
            'goal': goal,
            'method': best_method,
            'strength': strength,
            'context_match': best_match,
        })

        return ExecutionNudge(
            method=best_method,
            method_index=best_idx,
            strength=strength,
            reason="learned",
            context_match=best_match,
            stochastic_flag=False,
        )

    # ── Failure mode inference ────────────────────────────────────────────────

    def _infer_failure_mode(
        self,
        success: bool,
        outcome_value: float,
        pred_error: Optional[float],
        stats: MethodStats,
    ) -> str:
        """
        Infer failure mode from outcome and prediction error.

        Priority order:
        1. prediction_error (most reliable — direct causal signal)
        2. stochastic variance (statistical signal)
        3. outcome magnitude (weakest — heuristic)

        The fundamental split:
          large negative pred_error → the IDEA was wrong (prediction failure)
          near-zero pred_error      → the IDEA was right, EXECUTION failed
        """
        if success:
            return "success"

        # ── Prediction error attribution (most reliable) ──────────────────
        if pred_error is not None:
            if pred_error < -0.5:
                # Dreamer predicted success, got failure → wrong hypothesis
                return "prediction"
            elif abs(pred_error) < 0.15:
                # Dreamer was approximately right → execution failed
                # Distinguish timing vs incomplete by outcome magnitude
                if -0.1 < outcome_value < 0.1:
                    return "incomplete"  # Right direction, wrong magnitude
                else:
                    return "timing"      # Right idea, wrong moment

        # ── Stochastic variance ───────────────────────────────────────────
        if len(stats.recent_outcomes) >= 5:
            variance = float(np.var(list(stats.recent_outcomes)))
            if variance > STOCHASTIC_THRESHOLD:
                return "stochastic"

        # ── Outcome magnitude fallback ────────────────────────────────────
        if -0.1 < outcome_value < 0.1:
            return "incomplete"
        if outcome_value < -0.5:
            return "prediction"  # Strong negative without pred_error data
        return "timing"  # Default execution failure

    # ── Context matching ──────────────────────────────────────────────────────

    def _context_match(self, state: np.ndarray, goal: str) -> float:
        """
        Context match score from the goal's shared SNN.

        Returns the SNN's mean spike rate on the current state,
        normalised by its running EMA. Low on novel states (homeostatic
        thresholds not calibrated), higher on familiar states.

        Falls back to 0.5 (neutral) if SNN not yet created for this goal.
        """
        snn = self._goal_snns.get(goal)
        if snn is None:
            return 0.5
        return snn.process(self._ensure_state(state), reward=0.0)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _ensure_state(self, state) -> np.ndarray:
        """Ensure state is a float32 numpy array of the right dimension."""
        if not isinstance(state, np.ndarray):
            state = np.zeros(self.state_dim, dtype=np.float32)
        state = state.astype(np.float32)
        if state.size < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.size))
        return state[:self.state_dim]

    def get_failure_breakdown(self, goal: str) -> Dict[str, Dict]:
        """
        Return failure mode breakdown for all methods under a goal.

        Useful for logging and debugging — shows which methods fail
        and why, without exposing raw statistics that could be over-fitted.
        """
        result = {}
        for (g, method), stats in self._stats.items():
            if g != goal:
                continue
            total = stats.n_attempts
            if total == 0:
                continue
            result[method] = {
                'n_attempts': total,
                'ema_success': round(stats.ema_success, 3),
                'ema_variance': round(stats.ema_variance, 3),
                'failure_modes': dict(stats.failure_modes),
                'dominant_failure': max(
                    stats.failure_modes, key=stats.failure_modes.get
                ) if stats.failure_modes else 'none',
            }
        return result

    def summary(self) -> str:
        """Human-readable summary of profiler state."""
        n_goals = len({g for (g, _) in self._stats})
        n_methods = len(self._stats)
        lines = [
            f"ExecutionProfiler:",
            f"  Goals tracked: {n_goals}",
            f"  (goal, method) pairs: {n_methods}",
            f"  Nudges issued:   {self._nudges_issued}",
            f"  Nudges deferred: {self._nudges_deferred}",
            f"    of which novelty:     {self._novelty_deferrals}",
            f"    of which stochastic:  {self._stochastic_deferrals}",
        ]
        # Top methods per goal
        for (goal, method), s in sorted(self._stats.items()):
            if s.n_attempts >= MIN_ATTEMPTS:
                lines.append(
                    f"  {goal}/{method}: "
                    f"success={s.ema_success:.0%} "
                    f"var={s.ema_variance:.3f} "
                    f"n={s.n_attempts}"
                )
        return "\n".join(lines)
