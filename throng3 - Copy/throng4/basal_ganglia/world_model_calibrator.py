"""
world_model_calibrator.py
=========================
Tracks WorldModel prediction accuracy against real ALE transitions.

Usage in training loop:
    calibrator = WorldModelCalibrator(world_model, state_encoder)
    calibrator.reset_episode()

    # After each real env.step():
    calibrator.record(state_ram, action, next_state_ram)

    # Periodically:
    stats = calibrator.stats()
    print(stats)  # {'mae': 0.023, 'cosine': 0.94, 'confidence': 0.87, ...}

The .confidence property (0–1) indicates how much to trust synthetic rollouts:
    < 0.5  → WorldModel not calibrated, don't use synthetic steps
    0.5–0.7 → Use with caution (3–5 synthetic steps per real step)
    > 0.7  → WorldModel reliable (10–20 synthetic steps per real step)

Metrics tracked:
    mae         Mean absolute error between predicted and real next state
    cosine_sim  Cosine similarity (1.0 = perfect direction match)
    reward_mae  Error in predicted reward vs actual reward
    confidence  Composite score in [0, 1]
    n_samples   Rolling window size used
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


class WorldModelCalibrator:
    """
    Measures WorldModel prediction quality against real ALE transitions.

    Args:
        world_model:    A WorldModel instance (must have .predict(state, action)).
        state_encoder:  CompressedStateEncoder used to encode raw RAM obs.
        window:         Rolling window size for statistics (default 200 steps).
        log_path:       Optional jsonl file to write calibration records.
        log_interval:   How often (in real steps) to write a log entry.
    """

    def __init__(
        self,
        world_model,
        state_encoder,
        window: int = 200,
        log_path: Optional[Path] = None,
        log_interval: int = 50,
    ) -> None:
        self._wm      = world_model
        self._enc     = state_encoder
        self._window  = window
        self._log_path     = log_path
        self._log_interval = log_interval

        # Rolling error buffers
        self._mae_buf:    collections.deque = collections.deque(maxlen=window)
        self._cosine_buf: collections.deque = collections.deque(maxlen=window)
        self._rew_mae_buf:collections.deque = collections.deque(maxlen=window)

        self._step_count = 0   # total real steps recorded
        self._ep_count   = 0

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Call at the start of each episode."""
        self._ep_count += 1

    def record(
        self,
        state_ram:      np.ndarray,   # raw RAM before step  (128 uint8)
        action:         int,
        next_state_ram: np.ndarray,   # raw RAM after step   (128 uint8)
        real_reward:    float = 0.0,
    ) -> dict:
        """
        Compare WorldModel prediction vs real transition.
        Returns a dict with per-step metrics (also appended to rolling buffers).
        """
        # Encode both states
        s_enc  = self._enc.encode(state_ram.astype(np.float32) / 255.0).data
        ns_enc = self._enc.encode(next_state_ram.astype(np.float32) / 255.0).data

        # WorldModel prediction
        pred_ns, pred_rew = self._wm.predict(s_enc, action)

        # Metrics
        mae       = float(np.mean(np.abs(pred_ns - ns_enc)))
        cos_sim   = _cosine_sim(pred_ns, ns_enc)
        rew_mae   = abs(pred_rew - real_reward)

        self._mae_buf.append(mae)
        self._cosine_buf.append(cos_sim)
        self._rew_mae_buf.append(rew_mae)
        self._step_count += 1

        result = {
            "mae": mae, "cosine": cos_sim, "rew_mae": rew_mae,
        }

        if self._log_path and self._step_count % self._log_interval == 0:
            self._write_log(result)

        return result

    def stats(self) -> dict:
        """Return current rolling statistics as a dict."""
        if not self._mae_buf:
            return {"mae": None, "cosine": None, "rew_mae": None,
                    "confidence": 0.0, "n_samples": 0}

        mae     = float(np.mean(self._mae_buf))
        cosine  = float(np.mean(self._cosine_buf))
        rew_mae = float(np.mean(self._rew_mae_buf))

        confidence = _compute_confidence(mae, cosine, rew_mae)

        return {
            "mae":        round(mae, 4),
            "cosine":     round(cosine, 4),
            "rew_mae":    round(rew_mae, 4),
            "confidence": round(confidence, 3),
            "n_samples":  len(self._mae_buf),
            "step":       self._step_count,
        }

    @property
    def confidence(self) -> float:
        """Quick scalar confidence score in [0, 1]. 0 = uncalibrated."""
        s = self.stats()
        return s["confidence"]

    def recommended_dyna_ratio(self) -> int:
        """
        How many synthetic steps to run per real step.
        0 = don't use model yet, 20 = model very reliable.
        """
        c = self.confidence
        if c < 0.50: return 0
        if c < 0.60: return 3
        if c < 0.70: return 7
        if c < 0.80: return 12
        return 20

    def summary_line(self) -> str:
        """One-line string for console output."""
        s = self.stats()
        if s["mae"] is None:
            return "[calib] no data yet"
        return (f"[calib] mae={s['mae']:.4f}  cos={s['cosine']:.3f}  "
                f"rew_err={s['rew_mae']:.3f}  conf={s['confidence']:.2f}  "
                f"dyna={self.recommended_dyna_ratio()}x  n={s['n_samples']}")

    # ── Private ────────────────────────────────────────────────────────

    def _write_log(self, step_result: dict) -> None:
        stats = self.stats()
        record = {
            "t":        time.time(),
            "step":     self._step_count,
            "episode":  self._ep_count,
            **stats,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Helpers ────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors. 1.0 = identical direction."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_confidence(mae: float, cosine: float, rew_mae: float) -> float:
    """
    Composite confidence score in [0, 1].

    Weights:
        70% from state cosine similarity (direction of transition)
        20% from state MAE (magnitude accuracy)
        10% from reward MAE

    Thresholds calibrated empirically — adjust as data accumulates.
    """
    # Cosine: 1.0 = perfect, 0.0 = orthogonal → map to [0, 1]
    cos_score = max(0.0, cosine)

    # MAE: sigmoid-like decay. MAE ~0 = 1.0, MAE ~0.05 = 0.5, MAE ~0.2 = 0.1
    mae_score = float(np.exp(-mae / 0.03))

    # Reward MAE: 0 = perfect, cap penalty at 1.0
    rew_score = max(0.0, 1.0 - rew_mae)

    return round(0.70 * cos_score + 0.20 * mae_score + 0.10 * rew_score, 4)
