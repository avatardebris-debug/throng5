"""
decision_trace.py — Per-step record of what influenced action decisions.

Captures the full decision context: which brain region sourced the action,
Q-values considered, vetoes applied, timing breakdown, and influence signals.

Usage:
    trace = DecisionTrace(step=42, action_taken=3, action_source="striatum")
    trace.striatum_q_values = [0.1, 0.3, 0.8, 0.2]
    trace.causal_vetoes = [1]  # Action 1 was blocked
    logger.record(trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DecisionTrace:
    """Per-step record of what influenced the action decision."""

    # ── Core decision ──────────────────────────────────────────────────
    step: int
    action_taken: int
    action_source: str = "unknown"      # "striatum", "heuristic", "plan", "skill"

    # ── Candidates ─────────────────────────────────────────────────────
    striatum_action: int = 0
    striatum_q_values: Optional[List[float]] = None
    plan_action: Optional[int] = None        # SubgoalPlanner recommendation
    heuristic_action: Optional[int] = None   # Motor Cortex proven chain

    # ── Vetoes and overrides ───────────────────────────────────────────
    causal_vetoes: List[int] = field(default_factory=list)  # Actions blocked
    dead_end_detected: bool = False
    rehearsal_override: bool = False
    entropy_override: bool = False           # Epsilon forced by EntropyMonitor

    # ── Influence signals ──────────────────────────────────────────────
    dream_action_bias: Optional[List[float]] = None  # Basal Ganglia dream values
    threat_score: float = 0.0                # Amygdala threat level
    curiosity_bonus: float = 0.0             # Intrinsic reward
    surprise: float = 0.0                    # WM prediction error

    # ── Context ────────────────────────────────────────────────────────
    reward: float = 0.0
    episode_reward_so_far: float = 0.0
    epsilon: float = 0.0

    # ── Timing ─────────────────────────────────────────────────────────
    total_step_ms: float = 0.0
    region_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONL logging (numpy-safe)."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                d[k] = float(v)
            else:
                d[k] = v
        return d
