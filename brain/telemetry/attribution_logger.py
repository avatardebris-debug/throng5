"""
attribution_logger.py — Accumulate DecisionTraces and write episode summaries.

Provides diagnostic visibility into which brain regions are actually
helping vs hurting, where time is spent, and how the agent's strategy
evolves over episodes.

Usage:
    logger = AttributionLogger(session_logger)
    logger.record(trace)              # Every step
    logger.episode_summary(episode=1) # On episode end
    diag = logger.diagnostics()       # Cross-episode analysis
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from brain.telemetry.decision_trace import DecisionTrace


class AttributionLogger:
    """
    Accumulates decision traces and produces diagnostic summaries.

    Per-episode:
      - Action source distribution (striatum vs heuristic vs plan)
      - Veto frequency (how often causal model blocked actions)
      - Time allocation (% in each brain region)
      - Surprise correlation with reward

    Cross-episode:
      - Trends in action source (is the planner taking over?)
      - Death-correlated regions (which modules were active before death?)
      - Performance attribution (reward per action source)
    """

    def __init__(self, log_dir: Optional[str] = None, max_episode_traces: int = 5000):
        self._current_traces: List[DecisionTrace] = []
        self._max_traces = max_episode_traces

        # Cross-episode accumulators
        self._episode_summaries: deque = deque(maxlen=200)
        self._source_reward: Dict[str, List[float]] = defaultdict(list)
        self._veto_counts: Dict[str, int] = defaultdict(int)
        self._total_steps: int = 0
        self._total_episodes: int = 0

        # Log file
        self._log_path: Optional[Path] = None
        if log_dir:
            self._log_path = Path(log_dir) / "attribution.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: DecisionTrace) -> None:
        """Record a single step's decision trace."""
        self._current_traces.append(trace)
        self._total_steps += 1

        # Track vetoes
        if trace.causal_vetoes:
            self._veto_counts["causal"] += 1
        if trace.dead_end_detected:
            self._veto_counts["dead_end"] += 1
        if trace.rehearsal_override:
            self._veto_counts["rehearsal"] += 1
        if trace.entropy_override:
            self._veto_counts["entropy"] += 1

        # Prune if too many traces in one episode
        if len(self._current_traces) > self._max_traces:
            self._current_traces.pop(0)

    def episode_summary(self, episode: int, episode_reward: float) -> Dict[str, Any]:
        """
        Generate and store per-episode diagnostic summary.

        Call on _on_episode_done().
        """
        traces = self._current_traces
        if not traces:
            return {"episode": episode, "steps": 0}

        self._total_episodes += 1

        # Action source distribution
        source_counts: Dict[str, int] = defaultdict(int)
        for t in traces:
            source_counts[t.action_source] += 1

        total = len(traces)
        source_pcts = {k: round(v / total * 100, 1) for k, v in source_counts.items()}

        # Timing analysis
        region_totals: Dict[str, float] = defaultdict(float)
        for t in traces:
            for region, ms in t.region_times.items():
                region_totals[region] += ms
        total_time = sum(region_totals.values())
        time_pcts = {
            k: round(v / total_time * 100, 1) if total_time > 0 else 0
            for k, v in region_totals.items()
        }

        # Surprise stats
        surprises = [t.surprise for t in traces if t.surprise > 0]
        avg_surprise = float(np.mean(surprises)) if surprises else 0.0

        # Threat stats  
        threats = [t.threat_score for t in traces]
        max_threat = max(threats) if threats else 0.0

        # Veto stats for this episode
        episode_vetoes = sum(1 for t in traces if t.causal_vetoes)
        episode_dead_ends = sum(1 for t in traces if t.dead_end_detected)

        # Reward per action source  
        for t in traces:
            self._source_reward[t.action_source].append(t.reward)

        summary = {
            "episode": episode,
            "steps": total,
            "episode_reward": round(episode_reward, 4),
            "action_sources": source_pcts,
            "time_allocation": time_pcts,
            "avg_surprise": round(avg_surprise, 4),
            "max_threat": round(max_threat, 4),
            "causal_vetoes": episode_vetoes,
            "dead_end_detections": episode_dead_ends,
            "avg_curiosity": round(
                float(np.mean([t.curiosity_bonus for t in traces])), 4
            ),
        }

        self._episode_summaries.append(summary)

        # Write to log file
        if self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(summary) + "\n")
            except Exception:
                pass

        # Reset for next episode
        self._current_traces = []

        return summary

    def diagnostics(self) -> Dict[str, Any]:
        """
        Cross-episode diagnostic analysis.

        Shows trends, reward attribution, and module effectiveness.
        """
        if not self._episode_summaries:
            return {"status": "no_data"}

        summaries = list(self._episode_summaries)

        # Reward per action source
        source_avg_reward = {}
        for source, rewards in self._source_reward.items():
            if rewards:
                source_avg_reward[source] = round(float(np.mean(rewards)), 4)

        # Source trend (first half vs second half of episodes)
        half = len(summaries) // 2
        if half > 0:
            first_sources = defaultdict(list)
            second_sources = defaultdict(list)
            for s in summaries[:half]:
                for src, pct in s.get("action_sources", {}).items():
                    first_sources[src].append(pct)
            for s in summaries[half:]:
                for src, pct in s.get("action_sources", {}).items():
                    second_sources[src].append(pct)
            source_trends = {}
            all_sources = set(first_sources.keys()) | set(second_sources.keys())
            for src in all_sources:
                first_avg = np.mean(first_sources.get(src, [0]))
                second_avg = np.mean(second_sources.get(src, [0]))
                source_trends[src] = {
                    "early_pct": round(float(first_avg), 1),
                    "late_pct": round(float(second_avg), 1),
                    "trend": "increasing" if second_avg > first_avg * 1.1 else
                             "decreasing" if second_avg < first_avg * 0.9 else "stable",
                }
        else:
            source_trends = {}

        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "reward_per_source": source_avg_reward,
            "source_trends": source_trends,
            "veto_totals": dict(self._veto_counts),
            "recent_avg_surprise": round(
                float(np.mean([s["avg_surprise"] for s in summaries[-10:]])), 4
            ) if summaries else 0,
        }

    def report(self) -> Dict[str, Any]:
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "current_episode_traces": len(self._current_traces),
            "episode_summaries_stored": len(self._episode_summaries),
            "veto_totals": dict(self._veto_counts),
        }
