"""
amygdala_thalamus.py — Unified Threat Detection & Mode Gating Region.

Merges three previously separate components into a single coherent pipeline:
  1. ThreatEstimator (learned prediction: state → P(death within k steps))
  2. Amygdala (dream-based danger assessment from DreamerEngine output)
  3. ModeController (hysteresis-based EXPLORE/EXECUTE/SURVIVE switching)

The unified flow:
  perception → ThreatEstimator.predict() → base threat score
  dream results → Amygdala.assess() → dream-augmented threat
  combined threat → ModeController.update() → operating mode
  if CRITICAL → broadcast HALT to all higher regions via message bus

This resolves the throng3/4 redundancy where Amygdala and ThreatEstimator
operated independently, sometimes producing contradictory recommendations.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion


# ── Data types ────────────────────────────────────────────────────────

class ThreatLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    CRITICAL = "critical"


class OperatingMode(Enum):
    EXPLORE = "explore"     # High entropy, hypothesis-testing, epsilon=0.35
    EXECUTE = "execute"     # Current hypothesis, normal epsilon=0.15
    SURVIVE = "survive"     # Conservative, minimal exploration, epsilon=0.05


@dataclass
class ThreatAssessment:
    """Combined output from the unified threat pipeline."""
    threat_score: float               # 0-1, combined from estimator + dreams
    threat_level: ThreatLevel
    operating_mode: OperatingMode
    epsilon: float                    # Exploration rate for current mode
    should_halt_higher: bool          # True if prefrontal/slow processing should stop
    dream_danger: Optional[float]     # Dream-based threat (None if no dreams)
    estimator_threat: float           # Learned threat estimate
    reason: str
    n_catastrophic_dreams: int = 0


# ── Mode epsilon mapping ──────────────────────────────────────────────

MODE_EPSILON = {
    OperatingMode.EXPLORE: 0.35,
    OperatingMode.EXECUTE: 0.15,
    OperatingMode.SURVIVE: 0.05,
}


# ── Unified Brain Region ─────────────────────────────────────────────

class AmygdalaThalamus(BrainRegion):
    """
    Unified threat detection and mode gating brain region.

    Combines learned threat estimation, dream-based danger assessment,
    and hysteresis-controlled mode switching into a single pipeline.

    Fast path: runs every frame, < 1ms
    """

    def __init__(
        self,
        bus: MessageBus,
        # Threat estimator params
        n_features: int = 18,
        hidden_size: int = 32,
        threat_threshold: float = 0.60,
        # Amygdala params
        catastrophe_threshold: float = -1.0,
        danger_threshold: float = -0.5,
        surprise_weight: float = 0.3,
        # Mode controller params
        enter_survive: float = 0.60,
        exit_survive: float = 0.35,
        enter_explore: float = 0.20,
        hysteresis_steps: int = 5,
    ):
        super().__init__("amygdala_thalamus", bus)

        # ── Threat Estimator (learned NN) ─────────────────────────────
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.threat_threshold = threat_threshold
        rng = np.random.RandomState(42)

        # 2-layer NN: Linear(n_features → hidden) → ReLU → Linear(hidden → 1) → Sigmoid
        self._W1 = rng.randn(n_features, hidden_size).astype(np.float32) * 0.1
        self._b1 = np.zeros(hidden_size, dtype=np.float32)
        self._W2 = rng.randn(hidden_size, 1).astype(np.float32) * 0.1
        self._b2 = np.zeros(1, dtype=np.float32)
        self._train_steps = 0

        # ── Amygdala (dream assessment) ───────────────────────────────
        self.catastrophe_threshold = catastrophe_threshold
        self.danger_threshold = danger_threshold
        self.surprise_weight = surprise_weight
        self._alertness = 0.0
        self._override_cooldown = 0
        self._last_override_step = -100
        self._danger_history: deque = deque(maxlen=50)

        # ── Mode Controller (hysteresis) ──────────────────────────────
        self._mode = OperatingMode.EXECUTE
        self._enter_survive = enter_survive
        self._exit_survive = exit_survive
        self._enter_explore = enter_explore
        self._hysteresis_steps = hysteresis_steps
        self._below_exit_count = 0
        self._mode_counts = {m: 0 for m in OperatingMode}
        self._transitions: List[tuple] = []

    # ── BrainRegion Interface ─────────────────────────────────────────

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one tick of threat assessment.

        Expected inputs:
            features: np.ndarray — state feature vector
            dream_results: Optional[List[dict]] — from DreamerEngine
            surprise_level: float — 0-1 from prediction error tracker
            step: int — current step count
        """
        features = inputs.get("features")
        dream_results = inputs.get("dream_results")
        surprise_level = inputs.get("surprise_level", 0.0)
        step = inputs.get("step", self._step_count)

        # Step 1: Learned threat estimation
        estimator_threat = self._predict_threat(features) if features is not None else 0.0

        # Step 2: Dream-based danger assessment
        dream_danger = None
        n_catastrophic = 0
        if dream_results:
            dream_danger, n_catastrophic = self._assess_dreams(
                dream_results, surprise_level
            )

        # Step 3: Combine signals (weighted average when both available)
        if dream_danger is not None:
            combined_threat = 0.6 * estimator_threat + 0.4 * dream_danger
        else:
            combined_threat = estimator_threat

        # Step 4: Update alertness (exponential moving average)
        self._alertness = 0.8 * self._alertness + 0.2 * combined_threat

        # Step 5: Determine threat level
        threat_level = self._classify_threat(combined_threat, n_catastrophic)

        # Step 6: Update operating mode (with hysteresis)
        prev_mode = self._mode
        self._update_mode(combined_threat, step)

        # Step 7: Check if we should halt higher functions
        should_halt = (
            threat_level in (ThreatLevel.DANGER, ThreatLevel.CRITICAL)
            and step - self._last_override_step > 20  # cooldown
        )

        if should_halt:
            self._last_override_step = step
            # Broadcast HALT to higher regions
            self.send(
                target="prefrontal_cortex",
                msg_type="halt",
                payload={"threat": combined_threat, "reason": "threat detected"},
                priority=Priority.EMERGENCY,
            )
            # Also halt hippocampus slow processing
            self.send(
                target="hippocampus",
                msg_type="halt",
                payload={"threat": combined_threat},
                priority=Priority.EMERGENCY,
            )

        # Build assessment
        assessment = ThreatAssessment(
            threat_score=combined_threat,
            threat_level=threat_level,
            operating_mode=self._mode,
            epsilon=MODE_EPSILON[self._mode],
            should_halt_higher=should_halt,
            dream_danger=dream_danger,
            estimator_threat=estimator_threat,
            reason=self._build_reason(threat_level, n_catastrophic),
            n_catastrophic_dreams=n_catastrophic,
        )

        # Broadcast threat assessment to all regions
        self.broadcast(
            msg_type="threat_assessment",
            payload={
                "threat_score": assessment.threat_score,
                "threat_level": assessment.threat_level.value,
                "operating_mode": assessment.operating_mode.value,
                "epsilon": assessment.epsilon,
                "should_halt_higher": assessment.should_halt_higher,
            },
            priority=Priority.URGENT if should_halt else Priority.ROUTINE,
        )

        return {
            "assessment": assessment,
            "threat_score": combined_threat,
            "operating_mode": self._mode.value,
            "epsilon": MODE_EPSILON[self._mode],
            "should_halt": should_halt,
        }

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the threat estimator from experience.

        Expected experience:
            X: np.ndarray — feature matrix (N, n_features)
            y: np.ndarray — labels (N,), 0.0=safe, 1.0=died
        """
        X = experience.get("X")
        y = experience.get("y")
        if X is None or y is None:
            return {"loss": 0.0}

        lr = experience.get("lr", 0.01)
        epochs = experience.get("epochs", 10)

        total_loss = 0.0
        for _ in range(epochs):
            # Forward
            hidden = np.maximum(0, X @ self._W1 + self._b1)  # ReLU
            logits = hidden @ self._W2 + self._b2
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits.flatten(), -10, 10)))  # Sigmoid

            # Binary cross-entropy loss
            eps = 1e-7
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            total_loss += loss

            # Backward
            dlogits = (probs - y).reshape(-1, 1)  # (N, 1)
            dW2 = hidden.T @ dlogits / len(y)
            db2 = np.mean(dlogits, axis=0)
            dhidden = dlogits @ self._W2.T
            dhidden[hidden <= 0] = 0  # ReLU grad
            dW1 = X.T @ dhidden / len(y)
            db1 = np.mean(dhidden, axis=0)

            self._W1 -= lr * dW1
            self._b1 -= lr * db1
            self._W2 -= lr * dW2
            self._b2 -= lr * db2

        self._train_steps += epochs
        return {"loss": total_loss / epochs, "train_steps": self._train_steps}

    def report(self) -> Dict[str, Any]:
        base = super().report()
        return {
            **base,
            "alertness": round(self._alertness, 3),
            "mode": self._mode.value,
            "epsilon": MODE_EPSILON[self._mode],
            "train_steps": self._train_steps,
            "mode_distribution": {m.value: c for m, c in self._mode_counts.items()},
            "recent_danger_ratio": self._recent_danger_ratio(),
        }

    # ── Internal: Threat Estimator ────────────────────────────────────

    def _predict_threat(self, features: np.ndarray) -> float:
        """Predict threat probability from state features."""
        x = np.asarray(features, dtype=np.float32).flatten()
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[:self.n_features]

        hidden = np.maximum(0, x @ self._W1 + self._b1)
        raw = hidden @ self._W2 + self._b2
        logit = float(np.asarray(raw).flatten()[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

    # ── Internal: Dream Assessment ────────────────────────────────────

    def _assess_dreams(
        self, dream_results: List[dict], surprise_level: float
    ) -> tuple:
        """Assess danger from dream simulation results. Returns (danger_score, n_catastrophic)."""
        if not dream_results:
            return 0.0, 0

        n_catastrophic = 0
        worst_reward = float("inf")
        all_negative = True

        for dr in dream_results:
            reward = dr.get("total_predicted_reward", 0.0)
            worst_step = dr.get("worst_step_reward", 0.0)

            if reward > 0:
                all_negative = False
            if worst_step < self.catastrophe_threshold:
                n_catastrophic += 1
            worst_reward = min(worst_reward, reward)

        # Base danger from dream results
        if all_negative and n_catastrophic > 0:
            danger = 0.9
        elif n_catastrophic > len(dream_results) // 2:
            danger = 0.7
        elif all_negative:
            danger = 0.5
        else:
            danger = 0.2

        # Modulate by surprise
        danger = danger * (1 + self.surprise_weight * surprise_level)
        danger = min(danger, 1.0)

        return danger, n_catastrophic

    # ── Internal: Mode Controller ─────────────────────────────────────

    def _update_mode(self, threat: float, step: int) -> None:
        """Update operating mode with hysteresis."""
        prev = self._mode

        if self._mode == OperatingMode.SURVIVE:
            if threat < self._exit_survive:
                self._below_exit_count += 1
            else:
                self._below_exit_count = 0

            if self._below_exit_count >= self._hysteresis_steps:
                self._mode = OperatingMode.EXECUTE
                self._below_exit_count = 0
                # Resume halted regions
                self.send("prefrontal_cortex", "resume", {}, Priority.URGENT)
                self.send("hippocampus", "resume", {}, Priority.URGENT)
        else:
            if threat >= self._enter_survive:
                self._mode = OperatingMode.SURVIVE
                self._below_exit_count = 0
            elif threat < self._enter_explore:
                self._mode = OperatingMode.EXPLORE
            else:
                self._mode = OperatingMode.EXECUTE

        self._mode_counts[self._mode] += 1
        if prev != self._mode:
            self._transitions.append((step, prev.value, self._mode.value, round(threat, 3)))

    # ── Internal: Classification ──────────────────────────────────────

    def _classify_threat(self, threat: float, n_catastrophic: int) -> ThreatLevel:
        """Classify combined threat score into a level."""
        if threat >= 0.8 or n_catastrophic >= 2:
            level = ThreatLevel.CRITICAL
        elif threat >= 0.6:
            level = ThreatLevel.DANGER
        elif threat >= 0.3:
            level = ThreatLevel.CAUTION
        else:
            level = ThreatLevel.SAFE

        self._danger_history.append(level)
        return level

    def _build_reason(self, level: ThreatLevel, n_catastrophic: int) -> str:
        if level == ThreatLevel.CRITICAL:
            return f"CRITICAL: {n_catastrophic} catastrophic dreams, alertness={self._alertness:.2f}"
        elif level == ThreatLevel.DANGER:
            return f"DANGER: High threat, alertness={self._alertness:.2f}"
        elif level == ThreatLevel.CAUTION:
            return f"CAUTION: Moderate threat"
        return "SAFE"

    def _recent_danger_ratio(self) -> float:
        if not self._danger_history:
            return 0.0
        danger_count = sum(
            1 for d in self._danger_history
            if d in (ThreatLevel.DANGER, ThreatLevel.CRITICAL)
        )
        return danger_count / len(self._danger_history)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._mode = OperatingMode.EXECUTE
        self._below_exit_count = 0
        self._alertness = 0.0

    def save_weights(self) -> Dict[str, np.ndarray]:
        return {"W1": self._W1, "b1": self._b1, "W2": self._W2, "b2": self._b2}

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self._W1 = weights["W1"]
        self._b1 = weights["b1"]
        self._W2 = weights["W2"]
        self._b2 = weights["b2"]
