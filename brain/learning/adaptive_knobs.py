"""
adaptive_knobs.py — Self-tuning controller for replay and exploration hyperparameters.

Core idea
---------
Every knob has a measurable feedback signal: did the learning improve after it
was used?  We track reward before/after each near-death replay trigger and use
that delta to adaptively adjust each knob.

This is gradient-free bandit hill climbing:
  - After each trigger: measure reward_delta = avg_reward_after - avg_reward_before
  - If improvement > threshold: nudge the knob further in the same direction
  - If regression: nudge back with a smaller step
  - Add small noise every `explore_every` updates to escape local optima

Knobs managed
-------------
  max_walk          : NearDeathReplayer — backward search depth
  bonus_reward      : NearDeathReplayer — synthetic escape reward
  near_death_thresh : NearDeathReplayer — reward threshold for near-death detection
  trigger_interval  : Orchestrator — episodes between ND replay triggers
  elite_halflife    : EliteReplayBuffer — decay halflife for elite fraction schedule
  elite_floor       : EliteReplayBuffer — floor fraction for elite sampling

Signals used
------------
  reward_delta      : avg(reward_last_N episodes after trigger) - avg(reward_last_N before)
  escape_rate       : fraction of triggers where a Q-guided escape was found
  elite_turnover    : fraction of elite slots that were replaced (diversity measure)

Adaptation logic per knob
--------------------------
  max_walk         ← escape_rate < 0.3 → increase; escape_rate > 0.7 → decrease
  bonus_reward     ← reward_delta positive → slowly decay bonus; negative → increase
  near_death_thresh← trigger never fires (escape_rate = 0 always) → lower threshold
  trigger_interval ← reward improving fast → increase interval; stagnant → decrease
  elite_halflife   ← reward_delta positive & stable → lengthen (slower decay)
                    reward_delta negative → shorten (faster shift to 20%)
  elite_floor      ← not used (left at 0.20 — empirically good default)

Usage
-----
    tuner = AdaptiveKnobController(window=20)
    tuner.attach(near_death_replayer, elite_buf, orchestrator)

    # After each episode:
    tuner.record_episode(episode_reward)

    # After each ND replay trigger fires:
    tuner.record_trigger()

    # Tuner auto-adjusts after every `tune_every` triggers
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class AdaptiveKnobController:
    """
    Self-tuning controller for NearDeathReplayer, EliteReplayBuffer, and
    orchestrator replay scheduling knobs.

    Parameters
    ----------
    window : int
        Episode window for computing before/after reward means.  Default 20.
    tune_every : int
        Number of ND-replay triggers between tuning updates.  Default 3.
    step_size : float
        Fractional step size for knob adjustments (0.10 = ±10%).  Default 0.10.
    explore_every : int
        Number of tuning steps before injecting random drift to escape
        local optima.  Default 10.
    verbose : bool
        Print tuning decisions to stdout.  Default False.
    """

    def __init__(
        self,
        window: int = 20,
        tune_every: int = 3,
        step_size: float = 0.10,
        explore_every: int = 10,
        verbose: bool = False,
    ):
        self._window = window
        self._tune_every = tune_every
        self._step = step_size
        self._explore_every = explore_every
        self._verbose = verbose

        # References to managed objects (set by attach())
        self._nd: Optional[Any] = None          # NearDeathReplayer
        self._elite: Optional[Any] = None       # EliteReplayBuffer
        self._orch: Optional[Any] = None        # WholeBrain orchestrator

        # Reward history
        self._episode_rewards: deque = deque(maxlen=window * 2)

        # Trigger tracking
        self._trigger_count = 0          # total triggers so far
        self._reward_at_trigger: list = []  # reward avg captured at each trigger

        # Tuning state
        self._tuning_steps = 0
        self._last_reward_delta = 0.0

        # Per-knob adjustment directions (for momentum)
        self._directions: Dict[str, int] = {
            "max_walk": 1, "bonus_reward": 1, "near_death_thresh": 1,
            "trigger_interval": 1, "elite_halflife": 1,
        }

        # Knob bounds (hard limits to avoid instability)
        self._bounds = {
            "max_walk":          (3, 40),
            "bonus_reward":      (0.2, 5.0),
            "near_death_thresh": (-5.0, -0.05),   # more negative = less sensitive
            "trigger_interval":  (2, 20),
            "elite_halflife":    (50.0, 1000.0),
        }

        # History of all adjustments for reporting
        self._adjustment_log: deque = deque(maxlen=200)

    # ── Attachment ────────────────────────────────────────────────────

    def attach(self, near_death_replayer, elite_buf, orchestrator) -> None:
        """Wire the controller to the managed objects."""
        self._nd = near_death_replayer
        self._elite = elite_buf
        self._orch = orchestrator

    # ── Observation API ───────────────────────────────────────────────

    def record_episode(self, episode_reward: float) -> None:
        """Call after every episode with the total reward."""
        self._episode_rewards.append(float(episode_reward))

    def record_trigger(self) -> None:
        """
        Call immediately after a ND replay trigger fires.

        Captures the current reward average as a pre-trigger snapshot.
        After `tune_every` triggers, runs the tuning update.
        """
        if len(self._episode_rewards) >= 2:
            recent_avg = float(np.mean(list(self._episode_rewards)[-self._window:]))
            self._reward_at_trigger.append(recent_avg)

        self._trigger_count += 1

        if self._trigger_count % self._tune_every == 0:
            self._tune()

    # ── Core tuning logic ─────────────────────────────────────────────

    def _tune(self) -> None:
        """Run one tuning step using the last `tune_every` trigger reward deltas."""
        if self._nd is None or self._elite is None or self._orch is None:
            return
        if len(self._reward_at_trigger) < 2:
            return

        # reward_delta = mean reward after trigger - mean reward before trigger
        n = min(self._tune_every, len(self._reward_at_trigger))
        recent_avgs = self._reward_at_trigger[-n:]
        reward_delta = recent_avgs[-1] - recent_avgs[0]
        self._last_reward_delta = reward_delta

        nd_report = self._nd.report()
        escape_rate = nd_report.get("escape_rate", 0.0)
        elite_report = self._elite.report()
        evictions = elite_report.get("evictions", 0)
        additions = elite_report.get("additions", 1)
        elite_turnover = evictions / max(additions, 1)

        self._tuning_steps += 1

        # ── Random exploration every N steps ──────────────────────────
        if self._tuning_steps % self._explore_every == 0:
            self._explore_drift()
            return

        # ── Adjust max_walk ────────────────────────────────────────────
        # If escape rate low → deeper search needed; if high → already effective
        if escape_rate < 0.25:
            self._nudge("max_walk", +1)
        elif escape_rate > 0.70:
            self._nudge("max_walk", -1)

        # ── Adjust bonus_reward ────────────────────────────────────────
        # If reward improving: bonus is working, gently tone it down to avoid
        # overriding the real reward signal. If stagnant: boost it.
        if reward_delta > 0.5:
            self._nudge("bonus_reward", -1)  # tone down synthetic reward
        elif reward_delta < -0.2:
            self._nudge("bonus_reward", +1)  # boost synthetic signal

        # ── Adjust near_death_thresh ───────────────────────────────────
        # If escape_rate is always 0 and no triggers find near-death states:
        # the threshold may be too strict (too negative). Raise it slightly.
        if escape_rate == 0.0 and nd_report.get("attempts", 0) > 5:
            self._nudge("near_death_thresh", +1)  # less negative = more sensitive
        elif escape_rate > 0.8:
            self._nudge("near_death_thresh", -1)  # stricter

        # ── Adjust trigger_interval ────────────────────────────────────
        # If reward improving fast: things are working, less frequent trigger OK.
        # If stagnant/regressing: trigger more often.
        if reward_delta > 1.0:
            self._nudge("trigger_interval", +1)   # less frequent
        elif reward_delta < 0.0:
            self._nudge("trigger_interval", -1)   # more frequent

        # ── Adjust elite halflife ──────────────────────────────────────
        # If elite buffer is turning over often (new bests found): lengthen halflife
        # so we stay in elite-dominated learning longer.
        # If stagnant (same 3 episodes forever): shorten so we reach 20% faster.
        if elite_turnover > 0.5:
            self._nudge("elite_halflife", +1)
        elif elite_turnover < 0.1 and reward_delta < 0.1:
            self._nudge("elite_halflife", -1)

        if self._verbose:
            print(
                f"[KnobTuner] step={self._tuning_steps} "
                f"Δreward={reward_delta:+.2f} escape={escape_rate:.2f} "
                f"| max_walk={self._nd.max_walk} bonus={self._nd.bonus_reward:.2f} "
                f"interval={self._orch._nd_trigger_interval}"
            )

    def _nudge(self, knob: str, direction: int) -> None:
        """
        Adjust a knob by step_size in `direction` (+1 or -1).
        Clips to bounds and writes to the managed object.
        """
        lo, hi = self._bounds[knob]
        current = self._get_knob(knob)
        if current is None:
            return

        # Fractional step for continuous knobs, integer step for discrete
        if isinstance(current, float):
            step = (hi - lo) * self._step * direction
            new_val = float(np.clip(current + step, lo, hi))
        else:
            new_val = int(np.clip(current + direction, lo, hi))

        self._set_knob(knob, new_val)
        self._adjustment_log.append({
            "step": self._tuning_steps,
            "knob": knob,
            "old": round(current, 4) if isinstance(current, float) else current,
            "new": round(new_val, 4) if isinstance(new_val, float) else new_val,
            "direction": direction,
            "reward_delta": round(self._last_reward_delta, 3),
        })

    def _explore_drift(self) -> None:
        """
        Inject small random perturbations to escape local optima.
        Each knob gets ±5% random drift.
        """
        for knob in self._bounds:
            if random.random() < 0.5:  # Only drift half the knobs per step
                direction = random.choice([-1, 1])
                lo, hi = self._bounds[knob]
                current = self._get_knob(knob)
                if current is None:
                    continue
                if isinstance(current, float):
                    drift = (hi - lo) * 0.05 * direction
                    new_val = float(np.clip(current + drift, lo, hi))
                else:
                    new_val = int(np.clip(current + direction, lo, hi))
                self._set_knob(knob, new_val)

        if self._verbose:
            print(f"[KnobTuner] step={self._tuning_steps} → explore drift applied")

    # ── Knob accessors (keeps the management logic in one place) ──────

    def _get_knob(self, knob: str):
        try:
            if knob == "max_walk":
                return self._nd.max_walk
            elif knob == "bonus_reward":
                return self._nd.bonus_reward
            elif knob == "near_death_thresh":
                return self._nd.near_death_thresh
            elif knob == "trigger_interval":
                return self._orch._nd_trigger_interval
            elif knob == "elite_halflife":
                return self._elite.halflife
        except AttributeError:
            return None

    def _set_knob(self, knob: str, value) -> None:
        try:
            if knob == "max_walk":
                self._nd.max_walk = value
            elif knob == "bonus_reward":
                self._nd.bonus_reward = value
            elif knob == "near_death_thresh":
                self._nd.near_death_thresh = value
            elif knob == "trigger_interval":
                self._orch._nd_trigger_interval = value
            elif knob == "elite_halflife":
                self._elite.halflife = value
        except AttributeError:
            pass

    # ── Reporting ─────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Full status report for telemetry."""
        snapshots = {}
        for knob in self._bounds:
            v = self._get_knob(knob)
            snapshots[knob] = round(v, 4) if isinstance(v, float) else v

        return {
            "tuning_steps": self._tuning_steps,
            "trigger_count": self._trigger_count,
            "last_reward_delta": round(self._last_reward_delta, 4),
            "current_knobs": snapshots,
            "recent_adjustments": list(self._adjustment_log)[-10:],
        }

    def __repr__(self) -> str:
        snaps = {k: self._get_knob(k) for k in self._bounds}
        return f"AdaptiveKnobController(step={self._tuning_steps}, knobs={snaps})"
