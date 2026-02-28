"""
heuristic_generator.py — Generate fast-path heuristics from dream analysis.

Takes successful episode patterns, dream simulations, and LLM strategy
suggestions and compresses them into simple if-then rules that the
Motor Cortex can execute in <1ms.

This is how the overnight loop produces lasting improvements:
  Dream → Analysis → Heuristic rule → Motor Cortex fast lookup

The generated heuristics are:
  state_hash → preferred_action
  (with optional conditions and confidence)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Heuristic:
    """A single compiled heuristic rule."""
    state_hash: int
    preferred_action: int
    confidence: float          # 0-1
    source: str                # "replay", "dream", "llm"
    times_validated: int = 0   # How many times this was correct post-hoc
    times_used: int = 0


class HeuristicGenerator:
    """
    Generates fast-path heuristic rules from overnight analysis.

    Three sources of heuristics:
    1. Replay success patterns — states that consistently led to high reward
    2. Dream simulations — actions that performed well in imagination
    3. LLM suggestions — strategic rules from Prefrontal Cortex analysis
    """

    def __init__(self, max_heuristics: int = 5000, confidence_threshold: float = 0.6):
        self._heuristics: Dict[int, Heuristic] = {}
        self._max_heuristics = max_heuristics
        self._confidence_threshold = confidence_threshold

        # State-action success tracking
        self._success_counts: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._attempt_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        self._total_generated = 0

    # ── Generation from replay ────────────────────────────────────────

    def process_replay_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_actions: int = 18,
    ) -> int:
        """
        Process a batch of replayed transitions to extract heuristics.

        Tracks which actions succeed at which states and generates
        heuristic rules when confidence exceeds threshold.

        Returns number of new heuristics generated.
        """
        new_count = 0

        for state, action, reward in zip(states, actions, rewards):
            key = self._hash_state(state)
            self._success_counts[key][action] += max(reward, 0)
            self._attempt_counts[key][action] += 1

            # Check if any action has enough evidence
            total_attempts = sum(self._attempt_counts[key].values())
            if total_attempts >= 5:  # Minimum evidence threshold
                best_action = max(
                    self._success_counts[key],
                    key=lambda a: self._success_counts[key][a] / max(self._attempt_counts[key][a], 1)
                )
                best_rate = self._success_counts[key][best_action] / max(self._attempt_counts[key][best_action], 1)

                if best_rate > self._confidence_threshold:
                    if key not in self._heuristics or self._heuristics[key].confidence < best_rate:
                        self._heuristics[key] = Heuristic(
                            state_hash=key,
                            preferred_action=best_action,
                            confidence=min(best_rate, 1.0),
                            source="replay",
                        )
                        new_count += 1

        self._total_generated += new_count
        self._prune_if_needed()
        return new_count

    # ── Generation from dreams ────────────────────────────────────────

    def process_dream_results(
        self,
        dream_trajectories: List[Dict[str, Any]],
    ) -> int:
        """
        Extract heuristics from dream simulation trajectories.

        Each dream trajectory should have:
            states: List[np.ndarray]
            actions: List[int]
            rewards: List[float]
            total_reward: float
        """
        new_count = 0

        for dream in dream_trajectories:
            if dream.get("total_reward", 0) <= 0:
                continue  # Only learn from positive dreams

            states = dream.get("states", [])
            actions = dream.get("actions", [])
            total_reward = dream.get("total_reward", 0)

            for state, action in zip(states, actions):
                key = self._hash_state(state)
                confidence = min(total_reward / 10.0, 0.8)  # Dreams are less certain

                if key not in self._heuristics or self._heuristics[key].confidence < confidence:
                    self._heuristics[key] = Heuristic(
                        state_hash=key,
                        preferred_action=action,
                        confidence=confidence,
                        source="dream",
                    )
                    new_count += 1

        self._total_generated += new_count
        self._prune_if_needed()
        return new_count

    # ── Generation from LLM ──────────────────────────────────────────

    def add_llm_heuristic(
        self,
        state_pattern: np.ndarray,
        action: int,
        confidence: float = 0.7,
        description: str = "",
    ) -> None:
        """Add a heuristic rule from LLM analysis."""
        key = self._hash_state(state_pattern)
        self._heuristics[key] = Heuristic(
            state_hash=key,
            preferred_action=action,
            confidence=confidence,
            source="llm",
        )
        self._total_generated += 1

    # ── Lookup ────────────────────────────────────────────────────────

    def lookup(self, state: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Look up heuristic action for a state.

        Returns (action, confidence) or None if no heuristic matches.
        """
        key = self._hash_state(state)
        h = self._heuristics.get(key)
        if h is not None and h.confidence >= self._confidence_threshold:
            h.times_used += 1
            return h.preferred_action, h.confidence
        return None

    def validate(self, state: np.ndarray, actual_reward: float) -> None:
        """Record whether a heuristic prediction was validated by actual outcome."""
        key = self._hash_state(state)
        h = self._heuristics.get(key)
        if h is not None and actual_reward > 0:
            h.times_validated += 1
            # Boost confidence for validated heuristics
            h.confidence = min(h.confidence * 1.05, 1.0)
        elif h is not None and actual_reward < -1:
            # Degrade confidence for failed heuristics
            h.confidence *= 0.9
            if h.confidence < 0.3:
                del self._heuristics[key]

    # ── Export ────────────────────────────────────────────────────────

    def export_for_motor_cortex(self) -> Dict[int, int]:
        """Export heuristics as a simple hash→action lookup for Motor Cortex."""
        return {
            h.state_hash: h.preferred_action
            for h in self._heuristics.values()
            if h.confidence >= self._confidence_threshold
        }

    def stats(self) -> Dict[str, Any]:
        if not self._heuristics:
            return {"n_heuristics": 0, "total_generated": self._total_generated}
        confs = [h.confidence for h in self._heuristics.values()]
        by_source = defaultdict(int)
        for h in self._heuristics.values():
            by_source[h.source] += 1
        return {
            "n_heuristics": len(self._heuristics),
            "total_generated": self._total_generated,
            "avg_confidence": round(np.mean(confs), 3),
            "by_source": dict(by_source),
            "total_used": sum(h.times_used for h in self._heuristics.values()),
            "total_validated": sum(h.times_validated for h in self._heuristics.values()),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _hash_state(self, state: np.ndarray) -> int:
        """Quantize and hash state for fast lookup."""
        if state is None:
            return 0
        # Quantize to reduce collision but allow generalization
        quantized = np.round(np.asarray(state, dtype=np.float32) * 10).astype(np.int16)
        return hash(quantized.tobytes())

    def _prune_if_needed(self) -> None:
        """Remove lowest-confidence heuristics if over capacity."""
        if len(self._heuristics) > self._max_heuristics:
            sorted_h = sorted(self._heuristics.values(), key=lambda h: h.confidence, reverse=True)
            keep = set(h.state_hash for h in sorted_h[:self._max_heuristics])
            self._heuristics = {k: v for k, v in self._heuristics.items() if k in keep}
