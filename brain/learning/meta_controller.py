"""
meta_controller.py — Self-regulating bandit over learning algorithms.

Manages multiple competing learners and learns WHICH one to use.
Critically, it also tracks whether META-SELECTION ITSELF is useful,
and automatically collapses to a single learner when it isn't.

How it works:
  1. Maintains N learner slots (DQN, TorchDQN, etc.)
  2. Each slot tracks a rolling reward window
  3. A Thompson Sampling bandit allocates action-control probability
  4. Simultaneously, a "self-relevance" test compares:
     - Meta-controller performance (bandit-weighted selection)
     - Best single learner performance (what if we just used the best one?)
  5. If the difference is statistically insignificant after enough data,
     the meta-controller locks onto the best learner and SHUTS ITSELF OFF

Neuroscience parallel:
  Dorsomedial striatum (goal-directed) → active early, explores strategies
  Dorsolateral striatum (habitual)     → takes over once a strategy is proven
  This module transitions from exploratory to habitual automatically.

Usage:
    from brain.learning.meta_controller import MetaController

    meta = MetaController()
    meta.register_learner("numpy_dqn", numpy_dqn_instance)
    meta.register_learner("torch_dqn", torch_dqn_instance)

    # Each step:
    learner, name = meta.select_learner()
    action = learner.select_action(features)
    meta.report_reward(name, reward)

    # Periodically:
    if meta.is_collapsed:
        print(f"Meta-controller shut down, locked to: {meta.locked_learner}")
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LearnerSlot:
    """Tracks a registered learner and its performance."""
    name: str
    learner: Any                    # The actual learner object
    rewards: deque = field(default_factory=lambda: deque(maxlen=500))
    total_reward: float = 0.0
    total_selections: int = 0
    total_steps: int = 0

    # Thompson Sampling parameters (Beta distribution)
    alpha: float = 1.0              # Successes + prior
    beta_param: float = 1.0         # Failures + prior

    @property
    def mean_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards))

    @property
    def reward_std(self) -> float:
        if len(self.rewards) < 2:
            return float("inf")
        return float(np.std(self.rewards))


class MetaController:
    """
    Self-regulating meta-controller over multiple learners.

    Tracks its own value-add and automatically collapses
    when meta-selection isn't improving performance.
    """

    def __init__(
        self,
        relevance_window: int = 200,        # Episodes before self-evaluation
        collapse_threshold: float = 0.02,    # Meta must beat best by >2% to stay active
        min_trials_per_learner: int = 30,    # Min trials before considering collapse
        confidence_level: float = 0.95,      # Statistical confidence for collapse decision
        drift_window: int = 50,              # Window for drift detection while collapsed
        drift_threshold: float = 0.20,       # Performance drop % that triggers uncollapse
    ):
        self._slots: Dict[str, LearnerSlot] = {}
        self._selection_order: List[str] = []

        # ── Meta relevance tracking ───────────────────────────────────
        self._relevance_window = relevance_window
        self._collapse_threshold = collapse_threshold
        self._min_trials = min_trials_per_learner
        self._confidence = confidence_level

        # Performance of meta-controller vs best single learner
        self._meta_rewards: deque = deque(maxlen=relevance_window)
        self._best_single_rewards: deque = deque(maxlen=relevance_window)

        # ── Drift detector (lightweight watchdog while collapsed) ─────
        self._drift_window = drift_window
        self._drift_threshold = drift_threshold
        self._reward_at_collapse: float = 0.0     # Baseline reward when we collapsed
        self._post_collapse_rewards: deque = deque(maxlen=drift_window)

        # ── Self-tuning: remember initial values + accumulated bias ───
        self._init_relevance_window = relevance_window
        self._init_min_trials = min_trials_per_learner
        self._init_drift_threshold = drift_threshold
        self._init_drift_window = drift_window
        self._max_param_ratio = 5.0     # Reset if any param hits 5x original
        self._tuning_history: List[Dict[str, Any]] = []

        # ── State ─────────────────────────────────────────────────────
        self._is_collapsed = False
        self._locked_learner: Optional[str] = None
        self._collapse_reason: str = ""
        self._total_steps = 0
        self._collapse_step: int = 0
        self._collapse_count: int = 0             # How many times we've collapsed
        self._drift_uncollapse_count: int = 0     # How many were drift-triggered

        # ── Promotion thresholds ─────────────────────────────────────
        self._promotion_threshold = 0.50   # Win rate to trigger hard commit
        self._promotion_min_episodes = 30  # Min episodes before promotion

        # ── History for analysis ──────────────────────────────────────
        self._selection_history: deque = deque(maxlen=1000)
        self._relevance_scores: deque = deque(maxlen=100)

    # ── Learner Registration ──────────────────────────────────────────

    def register_learner(self, name: str, learner: Any) -> None:
        """Register a learner with the meta-controller."""
        self._slots[name] = LearnerSlot(name=name, learner=learner)
        self._selection_order.append(name)

    def unregister_learner(self, name: str) -> None:
        self._slots.pop(name, None)
        if name in self._selection_order:
            self._selection_order.remove(name)

    # ── Action Selection ──────────────────────────────────────────────

    def select_learner(self) -> Tuple[Any, str]:
        """
        Select which learner to use for this step.

        If collapsed, always returns the locked learner.
        Otherwise, uses Thompson Sampling to explore/exploit,
        with an initial round-robin phase to ensure all learners
        get minimum trials.
        """
        if not self._slots:
            raise ValueError("No learners registered")

        self._total_steps += 1

        # ── Collapsed: use locked learner ─────────────────────────────
        if self._is_collapsed and self._locked_learner:
            slot = self._slots[self._locked_learner]
            slot.total_selections += 1
            return slot.learner, slot.name

        # ── Forced exploration: round-robin until min_trials met ──────
        # Ensures every learner gets enough data for self-relevance check
        for name in self._selection_order:
            slot = self._slots[name]
            if slot.total_steps < self._min_trials:
                slot.total_selections += 1
                self._selection_history.append(name)
                return slot.learner, name

        # ── Thompson Sampling ─────────────────────────────────────────
        best_sample = -float("inf")
        best_name = self._selection_order[0]

        for name, slot in self._slots.items():
            # Sample from Beta distribution
            sample = np.random.beta(slot.alpha, slot.beta_param)
            if sample > best_sample:
                best_sample = sample
                best_name = name

        slot = self._slots[best_name]
        slot.total_selections += 1
        self._selection_history.append(best_name)
        return slot.learner, best_name

    # ── Reward Reporting ──────────────────────────────────────────────

    def report_reward(self, learner_name: str, reward: float) -> None:
        """
        Report reward received when using the named learner.

        Updates the bandit's prior and checks self-relevance.
        """
        if learner_name not in self._slots:
            return

        slot = self._slots[learner_name]
        slot.rewards.append(reward)
        slot.total_reward += reward
        slot.total_steps += 1

        # Update Thompson Sampling Beta distribution
        # Normalize reward to [0, 1] range for Beta distribution
        norm_reward = self._normalize_reward(reward)
        slot.alpha += norm_reward
        slot.beta_param += (1 - norm_reward)

        # Track meta-controller performance
        self._meta_rewards.append(reward)

        # Track what the best single learner would have gotten
        # (counterfactual: if we always used the currently-best learner)
        best_name = self._get_best_single_learner()
        if best_name and best_name in self._slots:
            best_slot = self._slots[best_name]
            if best_slot.rewards:
                self._best_single_rewards.append(best_slot.mean_reward)

        # ── Self-relevance check (when exploring) ─────────────────────
        if not self._is_collapsed:
            self._check_promotion()
            self._check_self_relevance()

        # ── Drift detector (when collapsed) ───────────────────────────
        if self._is_collapsed:
            self._check_drift(reward)

    # ── Self-Relevance System ─────────────────────────────────────────

    def _check_self_relevance(self) -> None:
        """
        Am I (the meta-controller) actually helping?

        Compare meta-selection performance vs. best-single-learner.
        If statistically indistinguishable, collapse.
        """
        # Need minimum data
        if len(self._meta_rewards) < self._relevance_window:
            return

        all_have_min = all(
            slot.total_steps >= self._min_trials
            for slot in self._slots.values()
        )
        if not all_have_min:
            return

        # ── Compute relevance score ───────────────────────────────────
        meta_mean = float(np.mean(self._meta_rewards))
        best_name = self._get_best_single_learner()

        if not best_name:
            return

        best_slot = self._slots[best_name]
        best_mean = best_slot.mean_reward

        # Avoid division by zero
        denom = max(abs(best_mean), 1e-6)
        relevance = (meta_mean - best_mean) / denom
        self._relevance_scores.append(relevance)

        # ── Statistical test ──────────────────────────────────────────
        # Check if meta-controller adds more than collapse_threshold
        if relevance < self._collapse_threshold:
            # Check if we're confident (multiple consecutive negative readings)
            recent_scores = list(self._relevance_scores)[-10:]
            if len(recent_scores) >= 5:
                avg_relevance = np.mean(recent_scores)
                if avg_relevance < self._collapse_threshold:
                    self._collapse(best_name, avg_relevance)

    def _check_promotion(self) -> None:
        """
        Fast-path: promote a learner with dominant win rate.

        If any learner's Thompson Sampling win rate exceeds the
        promotion threshold for enough episodes, commit immediately.
        """
        if self._is_collapsed:
            return

        for name, slot in self._slots.items():
            if slot.total_steps < self._promotion_min_episodes:
                continue
            win_rate = slot.alpha / max(slot.alpha + slot.beta_param, 1e-6)
            if win_rate > self._promotion_threshold:
                self._collapse(
                    name,
                    relevance_score=win_rate,
                )
                self._collapse_reason = (
                    f"promoted: {name} win_rate={win_rate:.3f} "
                    f"after {slot.total_steps} episodes"
                )
                return

    def _collapse(self, best_learner: str, relevance_score: float) -> None:
        """
        Shut down meta-selection and lock to the best learner.

        The meta-controller determined it wasn't adding value.
        Records baseline reward for drift detection.
        Tunes parameters based on how long this cycle took.
        """
        # ── Self-tune: adjust parameters based on this cycle ────────
        steps_to_collapse = self._total_steps - self._collapse_step
        self._tune_after_collapse(steps_to_collapse)

        self._is_collapsed = True
        self._locked_learner = best_learner
        self._collapse_step = self._total_steps
        self._collapse_count += 1
        self._collapse_reason = (
            f"Meta-selection relevance ({relevance_score:+.4f}) below "
            f"threshold ({self._collapse_threshold:+.4f}) after "
            f"{self._total_steps} steps. Locked to '{best_learner}'."
        )

        # Record baseline for drift detection
        slot = self._slots[best_learner]
        self._reward_at_collapse = slot.mean_reward
        self._post_collapse_rewards.clear()

    def _check_drift(self, reward: float) -> None:
        """
        Lightweight watchdog: is the locked learner still performing?

        Runs every step while collapsed. Cost: one deque append + one
        float comparison per step — effectively zero overhead.

        If performance drops by >drift_threshold vs. collapse baseline,
        auto-uncollapse to re-explore.
        """
        self._post_collapse_rewards.append(reward)

        # Need enough data
        if len(self._post_collapse_rewards) < self._drift_window:
            return

        current_mean = float(np.mean(self._post_collapse_rewards))
        baseline = self._reward_at_collapse

        # Avoid division by zero
        if abs(baseline) < 1e-6:
            # If baseline was ~0, check absolute drop
            if current_mean < -0.1:
                self._drift_uncollapse(current_mean, baseline)
            return

        drop = (baseline - current_mean) / abs(baseline)

        if drop > self._drift_threshold:
            self._drift_uncollapse(current_mean, baseline)

    def _drift_uncollapse(self, current_mean: float, baseline: float) -> None:
        """Auto-uncollapse due to performance drift."""
        old_learner = self._locked_learner
        self._collapse_reason = (
            f"DRIFT DETECTED: '{old_learner}' reward dropped from "
            f"{baseline:.4f} → {current_mean:.4f} "
            f"(>{self._drift_threshold:.0%} drop). Re-exploring."
        )
        self._drift_uncollapse_count += 1
        self._tune_after_drift()
        self.uncollapse()

    # ── Self-Tuning ───────────────────────────────────────────────

    def _tune_after_collapse(self, steps_this_cycle: int) -> None:
        """
        Adjust parameters after a successful collapse.

        If collapse was fast -> we had enough data, shrink window slightly.
        If collapse was slow -> we needed more data, grow window slightly.
        Bias accumulates across cycles, never resets (unless capped).
        """
        expected = self._relevance_window * 2  # rough expected cycle length

        if steps_this_cycle < expected * 0.5:
            # Collapsed faster than expected -> shrink window 10%
            self._relevance_window = max(
                20, int(self._relevance_window * 0.9)
            )
        elif steps_this_cycle > expected * 2:
            # Collapsed slower than expected -> grow window 10%
            self._relevance_window = int(self._relevance_window * 1.1)

        self._tuning_history.append({
            "event": "collapse",
            "step": self._total_steps,
            "cycle_steps": steps_this_cycle,
            "relevance_window": self._relevance_window,
            "drift_threshold": round(self._drift_threshold, 4),
            "min_trials": self._min_trials,
        })

        self._check_param_caps()

    def _tune_after_drift(self) -> None:
        """
        Adjust parameters after a drift-triggered uncollapse.

        Each drift = we were too lenient. Tighten the drift threshold
        and grow the relevance window (be more cautious next time).
        """
        # Tighten drift threshold by 10% (more sensitive)
        self._drift_threshold = max(0.05, self._drift_threshold * 0.9)

        # Grow relevance window by 15% (need more data before collapsing)
        self._relevance_window = int(self._relevance_window * 1.15)

        # Grow min_trials slightly (explore more before deciding)
        self._min_trials = int(self._min_trials * 1.1)

        self._tuning_history.append({
            "event": "drift_uncollapse",
            "step": self._total_steps,
            "relevance_window": self._relevance_window,
            "drift_threshold": round(self._drift_threshold, 4),
            "min_trials": self._min_trials,
        })

        self._check_param_caps()

    def _check_param_caps(self) -> None:
        """
        Safety valve: if any parameter has drifted >5x from its
        original value, reset ALL parameters to initial values.
        Prevents runaway growth from pathological environments.
        """
        ratios = [
            self._relevance_window / max(self._init_relevance_window, 1),
            self._min_trials / max(self._init_min_trials, 1),
            self._init_drift_threshold / max(self._drift_threshold, 0.001),  # inverted: smaller = more extreme
        ]

        if any(r > self._max_param_ratio for r in ratios):
            self._relevance_window = self._init_relevance_window
            self._min_trials = self._init_min_trials
            self._drift_threshold = self._init_drift_threshold
            self._drift_window = self._init_drift_window
            self._tuning_history.append({
                "event": "PARAM_RESET",
                "step": self._total_steps,
                "reason": f"Parameter ratio exceeded {self._max_param_ratio}x cap",
            })

    # ── Uncollapse ───────────────────────────────────────────────

    def uncollapse(self) -> None:
        """
        Re-enter exploration mode.

        Preserves tuned parameters (accumulated bias). Only resets
        the bandit priors and reward tracking for the new cycle.
        """
        self._is_collapsed = False
        self._locked_learner = None
        # NOTE: _collapse_reason is preserved for inspection
        self._meta_rewards.clear()
        self._best_single_rewards.clear()
        self._relevance_scores.clear()

        # Resize deques to match tuned window
        self._meta_rewards = deque(maxlen=self._relevance_window)
        self._best_single_rewards = deque(maxlen=self._relevance_window)

        # Reset Thompson Sampling priors (but keep reward history)
        for slot in self._slots.values():
            slot.alpha = 1.0
            slot.beta_param = 1.0

    # ── Internal ──────────────────────────────────────────────────────

    def _get_best_single_learner(self) -> Optional[str]:
        """Find the learner with the highest mean reward."""
        best_name = None
        best_mean = -float("inf")
        for name, slot in self._slots.items():
            if slot.rewards and slot.mean_reward > best_mean:
                best_mean = slot.mean_reward
                best_name = name
        return best_name

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] for Beta distribution."""
        # Use sigmoid for unbounded rewards
        return 1.0 / (1.0 + math.exp(-np.clip(reward, -10, 10)))

    # ── Properties & Reporting ────────────────────────────────────────

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    @property
    def locked_learner(self) -> Optional[str]:
        return self._locked_learner

    @property
    def is_active(self) -> bool:
        """Is the meta-controller actively exploring?"""
        return not self._is_collapsed

    def report(self) -> Dict[str, Any]:
        """Full status report."""
        learner_stats = {}
        for name, slot in self._slots.items():
            learner_stats[name] = {
                "mean_reward": round(slot.mean_reward, 4),
                "reward_std": round(slot.reward_std, 4) if slot.reward_std != float("inf") else "∞",
                "total_selections": slot.total_selections,
                "total_steps": slot.total_steps,
                "alpha": round(slot.alpha, 2),
                "beta": round(slot.beta_param, 2),
            }

        relevance_scores = list(self._relevance_scores)
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0.0

        # Drift info
        drift_info = {}
        if self._is_collapsed and self._post_collapse_rewards:
            current = float(np.mean(self._post_collapse_rewards))
            drift_info = {
                "baseline_reward": round(self._reward_at_collapse, 4),
                "current_reward": round(current, 4),
                "drift_pct": round(
                    (self._reward_at_collapse - current)
                    / max(abs(self._reward_at_collapse), 1e-6) * 100, 1
                ),
                "drift_threshold_pct": self._drift_threshold * 100,
            }

        return {
            "mode": "collapsed" if self._is_collapsed else "exploring",
            "locked_learner": self._locked_learner,
            "collapse_reason": self._collapse_reason,
            "collapse_count": self._collapse_count,
            "total_steps": self._total_steps,
            "collapse_step": self._collapse_step if self._is_collapsed else None,
            "learners": learner_stats,
            "avg_relevance": round(avg_relevance, 4),
            "recent_relevance": [round(r, 4) for r in relevance_scores[-5:]],
            "n_registered": len(self._slots),
            "drift": drift_info,
        }
