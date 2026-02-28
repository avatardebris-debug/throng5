"""
session_logger.py — Structured JSON telemetry for Throng 5.

Writes one JSON object per line to a session log file. Every significant
decision, branch point, region activation, or error is captured so that:

1. A fresh context window can reconstruct what happened.
2. The AI can detect when work has branched from core and recommend returning.
3. Debugging can trace exactly which component made which decision.

Usage:
    from brain.telemetry.session_logger import SessionLogger

    log = SessionLogger("throng5_session")
    log.event("phase1", "restructure", "Copied throng4 into brain/")
    log.branch("phase1", "montezuma_fix", "Fixing Montezuma room constants")
    log.decision("striatum", "learner_swap", {"from": "DQN", "to": "PPO"}, reason="PPO better for continuous")
    log.error("basal_ganglia", "dreamer_nan", {"step": 1042}, severity="warning")
    log.merge("phase1", "montezuma_fix", "Fix integrated, returning to core")
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Default log directory (relative to throng5 root)
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "telemetry"


class SessionLogger:
    """
    Append-only structured JSON logger.

    Each line is a self-contained JSON object with:
        timestamp, session_id, phase, region, event_type, data, branch_tag, severity
    """

    def __init__(
        self,
        session_name: str = "default",
        log_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.session_name = session_name

        log_path = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        log_path.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_path / f"{session_name}_{ts}_{self.session_id}.jsonl"
        self._file = open(self.log_file, "a", encoding="utf-8")

        # Track active branches for detecting divergence
        self._active_branches: Dict[str, dict] = {}
        self._event_count = 0

        # Write session header
        self._write({
            "event_type": "session_start",
            "phase": "init",
            "region": "system",
            "data": {
                "session_name": session_name,
                "session_id": self.session_id,
                "log_file": str(self.log_file),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    # ── Core logging methods ──────────────────────────────────────────

    def event(
        self,
        phase: str,
        event_type: str,
        message: str,
        region: str = "system",
        data: Optional[Dict[str, Any]] = None,
        severity: str = "info",
    ) -> None:
        """Log a general event."""
        self._write({
            "event_type": event_type,
            "phase": phase,
            "region": region,
            "message": message,
            "data": data or {},
            "severity": severity,
        })

    def decision(
        self,
        region: str,
        decision_type: str,
        data: Dict[str, Any],
        reason: str = "",
        phase: str = "runtime",
    ) -> None:
        """Log a decision made by a brain region or the system."""
        self._write({
            "event_type": "decision",
            "phase": phase,
            "region": region,
            "message": f"Decision: {decision_type}",
            "data": {**data, "decision_type": decision_type, "reason": reason},
            "severity": "info",
        })

    def error(
        self,
        region: str,
        error_type: str,
        data: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        phase: str = "runtime",
    ) -> None:
        """Log an error or warning."""
        self._write({
            "event_type": "error",
            "phase": phase,
            "region": region,
            "message": f"Error: {error_type}",
            "data": {**(data or {}), "error_type": error_type},
            "severity": severity,
        })

    # ── Branch tracking ───────────────────────────────────────────────

    def branch(
        self,
        phase: str,
        branch_name: str,
        reason: str,
        region: str = "system",
    ) -> None:
        """
        Mark that work is branching from the core path.

        This is critical for detecting when development has diverged
        and recommending a return to the main line.
        """
        self._active_branches[branch_name] = {
            "started_at": time.time(),
            "start_event": self._event_count,
            "phase": phase,
            "reason": reason,
        }
        self._write({
            "event_type": "branch_start",
            "phase": phase,
            "region": region,
            "message": f"⑂ Branch started: {branch_name}",
            "data": {"branch_name": branch_name, "reason": reason},
            "severity": "info",
            "branch_tag": branch_name,
        })

    def merge(
        self,
        phase: str,
        branch_name: str,
        summary: str,
        region: str = "system",
    ) -> None:
        """Mark that a branch has been merged back to core."""
        branch_info = self._active_branches.pop(branch_name, {})
        duration = time.time() - branch_info.get("started_at", time.time())
        events_in_branch = self._event_count - branch_info.get("start_event", self._event_count)

        self._write({
            "event_type": "branch_merge",
            "phase": phase,
            "region": region,
            "message": f"⑂ Branch merged: {branch_name}",
            "data": {
                "branch_name": branch_name,
                "summary": summary,
                "duration_sec": round(duration, 1),
                "events_in_branch": events_in_branch,
            },
            "severity": "info",
            "branch_tag": branch_name,
        })

    def get_active_branches(self) -> Dict[str, dict]:
        """Return currently active (un-merged) branches."""
        return dict(self._active_branches)

    def recommend_return_to_core(self) -> Optional[str]:
        """
        If branches have been open too long, recommend returning.

        Returns a recommendation string or None.
        """
        recommendations = []
        for name, info in self._active_branches.items():
            elapsed = time.time() - info["started_at"]
            events = self._event_count - info["start_event"]
            if elapsed > 3600 or events > 50:  # 1 hour or 50 events
                recommendations.append(
                    f"Branch '{name}' has been open for {elapsed/60:.0f}min "
                    f"({events} events). Consider merging back to core."
                )
        return "\n".join(recommendations) if recommendations else None

    # ── Region activation tracking ────────────────────────────────────

    def region_activated(
        self,
        region: str,
        inputs: Dict[str, Any],
        phase: str = "runtime",
    ) -> None:
        """Log that a brain region was activated with given inputs."""
        self._write({
            "event_type": "region_activated",
            "phase": phase,
            "region": region,
            "message": f"Region activated: {region}",
            "data": {"inputs_summary": {k: type(v).__name__ for k, v in inputs.items()}},
            "severity": "debug",
        })

    def region_output(
        self,
        region: str,
        outputs: Dict[str, Any],
        processing_ms: float = 0.0,
        phase: str = "runtime",
    ) -> None:
        """Log a brain region's output."""
        self._write({
            "event_type": "region_output",
            "phase": phase,
            "region": region,
            "message": f"Region output: {region} ({processing_ms:.1f}ms)",
            "data": {"outputs_summary": {k: type(v).__name__ for k, v in outputs.items()}, "processing_ms": processing_ms},
            "severity": "debug",
        })

    # ── Milestone tracking ────────────────────────────────────────────

    def milestone(
        self,
        phase: str,
        description: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a significant achievement or checkpoint."""
        self._write({
            "event_type": "milestone",
            "phase": phase,
            "region": "system",
            "message": f"★ {description}",
            "data": metrics or {},
            "severity": "info",
        })

    # ── Training metrics ──────────────────────────────────────────────

    def training_step(
        self,
        region: str,
        episode: int,
        step: int,
        metrics: Dict[str, float],
        phase: str = "training",
    ) -> None:
        """Log a training step. Only logged every N steps to avoid spam."""
        self._write({
            "event_type": "training_step",
            "phase": phase,
            "region": region,
            "message": f"Training ep={episode} step={step}",
            "data": {"episode": episode, "step": step, **metrics},
            "severity": "debug",
        })

    # ── Internal ──────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        """Write a record to the JSONL log file."""
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["session_id"] = self.session_id
        record["seq"] = self._event_count

        # Add active branch context
        if self._active_branches and "branch_tag" not in record:
            record["branch_tag"] = list(self._active_branches.keys())[-1]

        self._event_count += 1

        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            self._file.write(line + "\n")
            self._file.flush()
        except Exception as e:
            # Never crash on logging failure
            print(f"[SessionLogger] Write error: {e}")

    def close(self) -> None:
        """Close the log file."""
        self._write({
            "event_type": "session_end",
            "phase": "shutdown",
            "region": "system",
            "data": {"total_events": self._event_count, "active_branches": list(self._active_branches.keys())},
        })
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            if not self._file.closed:
                self.close()
        except Exception:
            pass
