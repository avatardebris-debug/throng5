"""
context_restorer.py — Reconstruct session state from telemetry logs.

Reads JSONL log files produced by SessionLogger and provides:
1. A summary of what happened (decisions, milestones, errors)
2. Active branch detection (work that diverged and hasn't returned)
3. Last-known state of each brain region
4. Recommendations for what to do next

Usage:
    from brain.telemetry.context_restorer import ContextRestorer

    restorer = ContextRestorer("logs/telemetry/")
    summary = restorer.latest_session_summary()
    print(summary)

    # Or restore from specific session
    summary = restorer.session_summary("abc123def456")
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "telemetry"


class ContextRestorer:
    """Read and summarize telemetry logs for context restoration."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR

    def list_sessions(self, limit: int = 10) -> List[Dict[str, str]]:
        """List recent session log files, newest first."""
        if not self.log_dir.exists():
            return []

        files = sorted(self.log_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        sessions = []
        for f in files[:limit]:
            # Parse first line for session metadata
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    first = json.loads(fp.readline())
                sessions.append({
                    "file": str(f),
                    "session_id": first.get("session_id", "unknown"),
                    "session_name": first.get("data", {}).get("session_name", f.stem),
                    "started_at": first.get("data", {}).get("started_at", ""),
                })
            except Exception:
                sessions.append({"file": str(f), "session_id": "error", "session_name": f.stem})
        return sessions

    def load_session(self, log_file: str | Path) -> List[Dict[str, Any]]:
        """Load all events from a session log file."""
        events = []
        with open(log_file, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def latest_session_summary(self) -> str:
        """Get a human-readable summary of the most recent session."""
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return "No session logs found."
        return self.session_summary_from_file(sessions[0]["file"])

    def session_summary_from_file(self, log_file: str | Path) -> str:
        """Generate a detailed summary from a specific log file."""
        events = self.load_session(log_file)
        if not events:
            return "Empty session log."

        return self._build_summary(events)

    def _build_summary(self, events: List[Dict[str, Any]]) -> str:
        """Build a structured summary from events."""
        lines = []

        # Session header
        start_event = events[0] if events else {}
        session_data = start_event.get("data", {})
        lines.append(f"# Session: {session_data.get('session_name', 'unknown')}")
        lines.append(f"Session ID: {start_event.get('session_id', 'unknown')}")
        lines.append(f"Started: {session_data.get('started_at', '?')}")
        lines.append(f"Total events: {len(events)}")
        lines.append("")

        # Milestones
        milestones = [e for e in events if e.get("event_type") == "milestone"]
        if milestones:
            lines.append("## Milestones")
            for m in milestones:
                lines.append(f"  - {m.get('message', '')}")
            lines.append("")

        # Decisions
        decisions = [e for e in events if e.get("event_type") == "decision"]
        if decisions:
            lines.append(f"## Decisions ({len(decisions)} total)")
            for d in decisions[-10:]:  # last 10
                data = d.get("data", {})
                lines.append(f"  - [{d.get('region', '?')}] {data.get('decision_type', '?')}: {data.get('reason', '')}")
            lines.append("")

        # Errors
        errors = [e for e in events if e.get("event_type") == "error"]
        if errors:
            lines.append(f"## Errors/Warnings ({len(errors)} total)")
            for e in errors[-5:]:
                lines.append(f"  - [{e.get('severity', '?')}] [{e.get('region', '?')}] {e.get('message', '')}")
            lines.append("")

        # Branch status
        branches_started = [e for e in events if e.get("event_type") == "branch_start"]
        branches_merged = {e.get("data", {}).get("branch_name") for e in events if e.get("event_type") == "branch_merge"}
        open_branches = [b for b in branches_started if b.get("data", {}).get("branch_name") not in branches_merged]

        if open_branches:
            lines.append("## ⚠️ OPEN BRANCHES (not merged back to core)")
            for b in open_branches:
                data = b.get("data", {})
                lines.append(f"  - **{data.get('branch_name', '?')}**: {data.get('reason', '')}")
            lines.append("")
            lines.append("  > Recommendation: Review these branches and merge back to core before proceeding.")
            lines.append("")

        # Region activity summary
        region_counts: Dict[str, int] = defaultdict(int)
        for e in events:
            if e.get("event_type") in ("region_activated", "region_output"):
                region_counts[e.get("region", "unknown")] += 1

        if region_counts:
            lines.append("## Region Activity")
            for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {region}: {count} activations")
            lines.append("")

        # Last event
        if events:
            last = events[-1]
            lines.append(f"## Last Event")
            lines.append(f"  Type: {last.get('event_type')}")
            lines.append(f"  Time: {last.get('timestamp')}")
            lines.append(f"  Message: {last.get('message', last.get('data', {}))}")

        return "\n".join(lines)

    def find_divergence_points(self, log_file: str | Path) -> List[Dict[str, Any]]:
        """
        Find points where work diverged from the intended path.

        Looks for:
        - Branches that stayed open too long
        - Consecutive errors suggesting a wrong direction
        - Decisions that contradicted previous decisions
        """
        events = self.load_session(log_file)
        divergences = []

        # Open branches
        branches_started = {}
        for e in events:
            if e.get("event_type") == "branch_start":
                name = e.get("data", {}).get("branch_name")
                branches_started[name] = e
            elif e.get("event_type") == "branch_merge":
                name = e.get("data", {}).get("branch_name")
                branches_started.pop(name, None)

        for name, start_event in branches_started.items():
            divergences.append({
                "type": "open_branch",
                "branch": name,
                "reason": start_event.get("data", {}).get("reason"),
                "timestamp": start_event.get("timestamp"),
                "recommendation": f"Branch '{name}' was never merged. Review and integrate or revert.",
            })

        # Consecutive errors (3+ in a row from same region)
        error_runs: Dict[str, int] = defaultdict(int)
        prev_region = None
        for e in events:
            if e.get("event_type") == "error" and e.get("severity") in ("error", "critical"):
                region = e.get("region", "unknown")
                if region == prev_region:
                    error_runs[region] += 1
                    if error_runs[region] >= 3:
                        divergences.append({
                            "type": "error_cluster",
                            "region": region,
                            "count": error_runs[region],
                            "timestamp": e.get("timestamp"),
                            "recommendation": f"Region '{region}' had {error_runs[region]} consecutive errors. Architecture issue likely.",
                        })
                else:
                    error_runs[region] = 1
                prev_region = region
            else:
                prev_region = None

        return divergences
