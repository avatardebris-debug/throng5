"""
Failure Profiler — Categorizes failures to improve concept formation.

Each failure gets three dimensions:
  1. Attribution: What kind of failure? (Mechanical, Strategic, Temporal, Unknown)
  2. Surprise: How unexpected? (from prediction error magnitude)
  3. Confidence: How sure are we about the attribution? (self-calibrated)

Calibration works by tracking whether post-attribution actions improve:
  - If we say "strategic" and a policy change helps → calibration up
  - If we say "temporal" and a timing change helps → calibration up
  - If our fix doesn't help → calibration down for that category
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


class FailureMode(Enum):
    """Categories of failure for concept formation."""
    MECHANICAL = "mechanical"  # Couldn't execute action (hit wall, invalid move)
    STRATEGIC = "strategic"    # Wrong policy (suboptimal choice)
    TEMPORAL = "temporal"      # Bad timing (right action, wrong time)
    UNKNOWN = "unknown"        # Profiler confused (fallback)


@dataclass
class FailureAnalysis:
    """Result of failure categorization with three dimensions."""
    # Attribution → what kind of failure
    mode: FailureMode
    
    # Confidence → how sure are we (calibrated interval)
    confidence: float          # Point estimate 0.0-1.0
    confidence_lower: float    # Lower bound of 80% interval
    confidence_upper: float    # Upper bound of 80% interval
    
    # Surprise → how unexpected (from prediction error)
    surprise: float            # 0.0-1.0 (0=expected failure, 1=total shock)
    
    # Evidence
    evidence: str              # Human-readable explanation
    evidence_signals: Dict = field(default_factory=dict)  # Raw signals


class FailureProfiler:
    """
    Categorizes failures using blind heuristics + prediction error context.
    
    Self-calibrating: tracks whether attributions lead to correct fixes.
    
    No game names. Infers failure mode from:
    - State change magnitude (did action have effect?)
    - Reward trajectory (improving, declining, flat?)
    - Action history (cyclic patterns, repetition?)
    - Prediction error magnitude (how surprised was the agent?)
    """
    
    # Thresholds for categorization
    MECHANICAL_STATE_CHANGE_THRESHOLD = 0.01
    STRATEGIC_REWARD_DECLINE_THRESHOLD = -0.1
    TEMPORAL_CYCLE_WINDOW = 10
    
    # Calibration
    CALIBRATION_DECAY = 0.95  # Exponential decay for old calibration data
    INITIAL_CALIBRATION = 0.5  # Start at 50% accuracy (no information)
    
    def __init__(self):
        self.failure_count = 0
        self.categorization_history: List[FailureAnalysis] = []
        
        # Self-calibration: tracks (attempts, successes) per failure mode
        # "success" = post-attribution action improved performance
        self._calibration: Dict[FailureMode, Dict[str, float]] = {
            mode: {'attempts': 0.0, 'successes': 0.0}
            for mode in FailureMode
        }
    
    def categorize_failure(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        recent_history: List[Dict],
        prediction_error: Optional[float] = None
    ) -> FailureAnalysis:
        """
        Categorize a failure transition.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received (typically negative or zero for failure)
            next_state: State after action
            recent_history: Last N transitions for context
            prediction_error: Optional prediction error magnitude (from tracker)
            
        Returns:
            FailureAnalysis with attribution, confidence interval, and surprise
        """
        self.failure_count += 1
        
        # ── Compute raw signals ──
        state_change = np.linalg.norm(next_state - state)
        state_change_norm = state_change / (np.linalg.norm(state) + 1e-8)
        
        reward_delta = 0.0
        if len(recent_history) >= 3:
            recent_rewards = [t.get('reward', 0) for t in recent_history[-3:]]
            reward_delta = reward - np.mean(recent_rewards)
        
        action_history_boost = 0.0
        if len(recent_history) >= self.TEMPORAL_CYCLE_WINDOW:
            same_action = [
                t for t in recent_history[-self.TEMPORAL_CYCLE_WINDOW:]
                if t.get('action') == action
            ]
            if same_action:
                past_avg = np.mean([t.get('reward', 0) for t in same_action])
                action_history_boost = past_avg - reward
        
        signals = {
            'state_change_norm': float(state_change_norm),
            'reward_delta': float(reward_delta),
            'action_history_boost': float(action_history_boost),
            'prediction_error': float(prediction_error) if prediction_error else 0.0,
        }
        
        # ── Score each mode ──
        scores = self._score_modes(signals)
        
        # ── Pick best mode ──
        best_mode = max(scores, key=scores.get)
        raw_confidence = scores[best_mode]
        
        # ── Calibrate confidence ──
        calibrated_conf, conf_lower, conf_upper = self._calibrate_confidence(
            best_mode, raw_confidence
        )
        
        # ── Compute surprise from prediction error ──
        surprise = self._compute_surprise(prediction_error, signals)
        
        # ── Build evidence string ──
        evidence = self._build_evidence(best_mode, signals, scores)
        
        analysis = FailureAnalysis(
            mode=best_mode,
            confidence=calibrated_conf,
            confidence_lower=conf_lower,
            confidence_upper=conf_upper,
            surprise=surprise,
            evidence=evidence,
            evidence_signals=signals,
        )
        
        self.categorization_history.append(analysis)
        return analysis
    
    def _score_modes(self, signals: Dict) -> Dict[FailureMode, float]:
        """
        Score each failure mode based on signals.
        
        Returns dict of mode → score (0-1), where higher = more likely.
        """
        sc = signals['state_change_norm']
        rd = signals['reward_delta']
        ahb = signals['action_history_boost']
        pe = signals['prediction_error']
        
        scores = {}
        
        # Mechanical: state barely changed
        # High prediction error + no state change = very mechanical
        mech_score = max(0, 1.0 - sc / self.MECHANICAL_STATE_CHANGE_THRESHOLD)
        if pe > 0:
            mech_score *= (1.0 + min(pe, 2.0) * 0.3)  # PE boosts if state frozen
        scores[FailureMode.MECHANICAL] = min(mech_score, 1.0)
        
        # Strategic: state changed, reward declined
        # High prediction error + reward decline = very strategic
        if sc > self.MECHANICAL_STATE_CHANGE_THRESHOLD and rd < self.STRATEGIC_REWARD_DECLINE_THRESHOLD:
            strat_score = min(abs(rd) / 2.0, 1.0)  # Scale by how much reward declined
            if pe > 0:
                strat_score = min(strat_score + pe * 0.2, 1.0)  # PE boosts
            scores[FailureMode.STRATEGIC] = strat_score
        else:
            scores[FailureMode.STRATEGIC] = 0.0
        
        # Temporal: action worked better before
        # Low prediction error + high action_history_boost = temporal
        if ahb > 0.1:
            temp_score = min(ahb / 2.0, 1.0)
            if pe is not None and pe < 0.5:
                temp_score *= 1.2  # Low PE = agent expected this, just timing
            scores[FailureMode.TEMPORAL] = min(temp_score, 1.0)
        else:
            scores[FailureMode.TEMPORAL] = 0.0
        
        # Unknown: nothing matched well
        max_known = max(scores[FailureMode.MECHANICAL],
                       scores[FailureMode.STRATEGIC],
                       scores[FailureMode.TEMPORAL])
        scores[FailureMode.UNKNOWN] = max(0.3 - max_known, 0.0)
        
        return scores
    
    def _calibrate_confidence(
        self, mode: FailureMode, raw_confidence: float
    ) -> Tuple[float, float, float]:
        """
        Calibrate raw confidence using historical accuracy.
        
        Returns (point_estimate, lower_80, upper_80).
        """
        cal = self._calibration[mode]
        n = cal['attempts']
        
        if n < 5:
            # Not enough data to calibrate — use raw with wide interval
            point = raw_confidence * self.INITIAL_CALIBRATION
            margin = 0.3
        else:
            # Empirical accuracy from past attributions
            accuracy = cal['successes'] / (n + 1e-8)
            # Blend raw score with empirical accuracy
            point = raw_confidence * accuracy
            # Wilson score interval (simplified)
            margin = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
        
        lower = max(0.0, point - margin)
        upper = min(1.0, point + margin)
        point = np.clip(point, 0.0, 1.0)
        
        return float(point), float(lower), float(upper)
    
    def _compute_surprise(
        self, prediction_error: Optional[float], signals: Dict
    ) -> float:
        """
        Compute how surprising this failure was.
        
        Uses prediction error if available, otherwise estimates from signals.
        """
        if prediction_error is not None and prediction_error > 0:
            # Direct from prediction error tracker
            # Normalize: PE of 0 = not surprised, PE of 2+ = very surprised
            return float(np.clip(prediction_error / 2.0, 0.0, 1.0))
        
        # Estimate surprise from signals
        # Large reward delta + large state change = surprising
        rd = abs(signals.get('reward_delta', 0))
        sc = signals.get('state_change_norm', 0)
        estimated = (rd * 0.6 + sc * 0.4) / 2.0
        return float(np.clip(estimated, 0.0, 1.0))
    
    def _build_evidence(
        self, mode: FailureMode, signals: Dict, scores: Dict
    ) -> str:
        """Build human-readable evidence string."""
        sc = signals['state_change_norm']
        rd = signals['reward_delta']
        pe = signals['prediction_error']
        
        parts = []
        if mode == FailureMode.MECHANICAL:
            parts.append(f"State barely changed (Δ={sc:.4f})")
        elif mode == FailureMode.STRATEGIC:
            parts.append(f"State changed (Δ={sc:.4f}) but reward declined ({rd:+.2f})")
        elif mode == FailureMode.TEMPORAL:
            parts.append(f"Action worked better before (boost={signals['action_history_boost']:.2f})")
        else:
            parts.append(f"Insufficient evidence for specific category")
        
        if pe > 0:
            parts.append(f"prediction_error={pe:.3f}")
        
        # Show competing modes
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.1:
            runner_up = sorted_scores[1]
            parts.append(f"runner-up: {runner_up[0].value} ({runner_up[1]:.2f})")
        
        return "; ".join(parts)
    
    def record_calibration_outcome(
        self, mode: FailureMode, fix_helped: bool
    ):
        """
        Record whether a fix based on our attribution actually helped.
        
        This is how we self-calibrate. Call this when:
        - We attributed "strategic" and a policy change was tried → did it help?
        - We attributed "temporal" and timing was adjusted → did it help?
        
        Args:
            mode: The failure mode we attributed
            fix_helped: Whether the corresponding fix helped performance
        """
        cal = self._calibration[mode]
        
        # Decay old data
        cal['attempts'] *= self.CALIBRATION_DECAY
        cal['successes'] *= self.CALIBRATION_DECAY
        
        # Add new observation
        cal['attempts'] += 1.0
        if fix_helped:
            cal['successes'] += 1.0
    
    def get_failure_distribution(self) -> Dict[FailureMode, int]:
        """Get count of each failure mode."""
        distribution = {mode: 0 for mode in FailureMode}
        for analysis in self.categorization_history:
            distribution[analysis.mode] += 1
        return distribution
    
    def get_dominant_failure_mode(self) -> Optional[FailureMode]:
        """Get the most common failure mode."""
        if not self.categorization_history:
            return None
        distribution = self.get_failure_distribution()
        return max(distribution, key=distribution.get)
    
    def get_calibration_accuracy(self) -> Dict[str, float]:
        """Get current calibration accuracy per mode."""
        result = {}
        for mode in FailureMode:
            cal = self._calibration[mode]
            if cal['attempts'] > 1:
                result[mode.value] = cal['successes'] / (cal['attempts'] + 1e-8)
            else:
                result[mode.value] = self.INITIAL_CALIBRATION
        return result
    
    def summary(self) -> str:
        """Human-readable summary of failure categorization."""
        if not self.categorization_history:
            return "No failures categorized yet"
        
        distribution = self.get_failure_distribution()
        total = len(self.categorization_history)
        calibration = self.get_calibration_accuracy()
        
        # Average confidence and surprise
        avg_conf = np.mean([a.confidence for a in self.categorization_history])
        avg_surprise = np.mean([a.surprise for a in self.categorization_history])
        
        lines = [
            f"Failure Analysis ({total} failures, "
            f"avg confidence={avg_conf:.2f}, avg surprise={avg_surprise:.2f}):",
        ]
        
        for mode in FailureMode:
            count = distribution[mode]
            if count == 0:
                continue
            pct = 100 * count / total
            cal = calibration.get(mode.value, 0.5)
            # Get avg confidence interval for this mode
            mode_analyses = [a for a in self.categorization_history if a.mode == mode]
            avg_lower = np.mean([a.confidence_lower for a in mode_analyses])
            avg_upper = np.mean([a.confidence_upper for a in mode_analyses])
            lines.append(
                f"  {mode.value:12s}: {count:3d} ({pct:5.1f}%) "
                f"conf=[{avg_lower:.2f}, {avg_upper:.2f}] "
                f"calibration={cal:.0%}"
            )
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset for new environment. Keeps calibration (transfers across envs)."""
        self.failure_count = 0
        self.categorization_history.clear()
        # NOTE: calibration is NOT reset — it transfers across environments


if __name__ == "__main__":
    """Test the failure profiler with confidence calibration."""
    print("=" * 60)
    print("FAILURE PROFILER TEST (with calibration)")
    print("=" * 60)
    
    profiler = FailureProfiler()
    
    # Test 1: Mechanical failure (state doesn't change)
    print("\n[Test 1] Mechanical failure (hit wall)")
    state = np.array([1.0, 2.0, 3.0])
    next_state = np.array([1.001, 2.001, 3.001])
    analysis = profiler.categorize_failure(
        state, 0, -1.0, next_state, [], prediction_error=0.8
    )
    print(f"  Attribution: {analysis.mode.value}")
    print(f"  Confidence:  {analysis.confidence:.2f} [{analysis.confidence_lower:.2f}, {analysis.confidence_upper:.2f}]")
    print(f"  Surprise:    {analysis.surprise:.2f}")
    print(f"  Evidence:    {analysis.evidence}")
    assert analysis.mode == FailureMode.MECHANICAL
    
    # Test 2: Strategic failure (state changes, reward declines)
    print("\n[Test 2] Strategic failure (wrong choice)")
    state = np.array([1.0, 2.0, 3.0])
    next_state = np.array([2.0, 3.0, 4.0])
    recent = [{'reward': 1.0}, {'reward': 1.5}, {'reward': 1.2}]
    analysis = profiler.categorize_failure(
        state, 1, 0.0, next_state, recent, prediction_error=1.5
    )
    print(f"  Attribution: {analysis.mode.value}")
    print(f"  Confidence:  {analysis.confidence:.2f} [{analysis.confidence_lower:.2f}, {analysis.confidence_upper:.2f}]")
    print(f"  Surprise:    {analysis.surprise:.2f}")
    print(f"  Evidence:    {analysis.evidence}")
    assert analysis.mode == FailureMode.STRATEGIC
    
    # Test 3: Temporal failure (action worked before)
    print("\n[Test 3] Temporal failure (bad timing)")
    state = np.array([1.0, 2.0, 3.0])
    next_state = np.array([2.0, 3.0, 4.0])
    recent = [
        {'action': 2, 'reward': 5.0},
        {'action': 1, 'reward': 1.0},
        {'action': 2, 'reward': 4.5},
        {'action': 0, 'reward': 0.5},
        {'action': 2, 'reward': 5.2},
    ]
    analysis = profiler.categorize_failure(
        state, 2, 0.5, next_state, recent, prediction_error=0.2
    )
    print(f"  Attribution: {analysis.mode.value}")
    print(f"  Confidence:  {analysis.confidence:.2f} [{analysis.confidence_lower:.2f}, {analysis.confidence_upper:.2f}]")
    print(f"  Surprise:    {analysis.surprise:.2f}")
    print(f"  Evidence:    {analysis.evidence}")
    
    # Test 4: Calibration feedback
    print("\n[Test 4] Self-calibration")
    profiler.record_calibration_outcome(FailureMode.MECHANICAL, fix_helped=True)
    profiler.record_calibration_outcome(FailureMode.MECHANICAL, fix_helped=True)
    profiler.record_calibration_outcome(FailureMode.MECHANICAL, fix_helped=False)
    profiler.record_calibration_outcome(FailureMode.STRATEGIC, fix_helped=True)
    profiler.record_calibration_outcome(FailureMode.STRATEGIC, fix_helped=False)
    profiler.record_calibration_outcome(FailureMode.STRATEGIC, fix_helped=False)
    
    accuracies = profiler.get_calibration_accuracy()
    print(f"  Calibration: {accuracies}")
    assert accuracies['mechanical'] > accuracies['strategic'], \
        "Mechanical should have higher accuracy (2/3 vs 1/3)"
    print("✅ Calibration tracks accuracy correctly")
    
    # Summary
    print(f"\n{profiler.summary()}")
    
    print("\n✅ Failure profiler test complete!")
