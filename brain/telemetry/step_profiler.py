"""
step_profiler.py — Lightweight per-component timing for the WholeBrain pipeline.

Measures wall-clock time per brain region per step, accumulates stats,
and reports average + percentage of total for each component.

Usage:
    from brain.telemetry.step_profiler import StepProfiler

    profiler = StepProfiler()

    profiler.start("sensory")
    # ... sensory cortex code ...
    profiler.stop("sensory")

    profiler.start("striatum")
    # ... striatum code ...
    profiler.stop("striatum")

    profiler.tick()  # Mark end of one full step

    print(profiler.report_str())
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class StepProfiler:
    """
    Zero-overhead-when-disabled timing profiler for brain regions.

    Accumulates per-component elapsed time and provides
    summary statistics (avg ms/step, % of total).
    """

    def __init__(self, enabled: bool = True, window: int = 200):
        self.enabled = enabled
        self.window = window

        # Accumulators
        self._totals: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._recent: Dict[str, List[float]] = defaultdict(list)

        # Per-step tracking
        self._current_start: Dict[str, float] = {}
        self._step_total: float = 0.0
        self._step_count: int = 0
        self._step_start: float = 0.0

    def start(self, component: str) -> None:
        """Start timing a component."""
        if not self.enabled:
            return
        self._current_start[component] = time.perf_counter()

    def stop(self, component: str) -> None:
        """Stop timing a component and accumulate."""
        if not self.enabled:
            return
        start = self._current_start.pop(component, None)
        if start is None:
            return
        elapsed = time.perf_counter() - start
        self._totals[component] += elapsed
        self._counts[component] += 1

        recent = self._recent[component]
        recent.append(elapsed)
        if len(recent) > self.window:
            recent.pop(0)

    def tick(self) -> None:
        """Mark the end of one full brain step."""
        if not self.enabled:
            return
        self._step_count += 1

    def step_start(self) -> None:
        """Mark the start of one full brain step (for total timing)."""
        if not self.enabled:
            return
        self._step_start = time.perf_counter()

    def step_end(self) -> None:
        """Mark the end of one full brain step (for total timing)."""
        if not self.enabled:
            return
        if self._step_start > 0:
            self._step_total += time.perf_counter() - self._step_start
        self._step_count += 1

    def report(self) -> Dict[str, Dict[str, float]]:
        """
        Return timing statistics as dict.

        For each component:
          total_ms: total elapsed (ms)
          avg_ms: average per call (ms)
          recent_avg_ms: average of recent window (ms)
          pct: percentage of total step time
          calls: number of calls
        """
        result = {}
        total_all = max(self._step_total, sum(self._totals.values()))

        for comp in sorted(self._totals.keys()):
            total = self._totals[comp]
            count = self._counts[comp]
            recent = self._recent[comp]
            recent_avg = sum(recent) / len(recent) if recent else 0

            result[comp] = {
                "total_ms": total * 1000,
                "avg_ms": (total / count * 1000) if count else 0,
                "recent_avg_ms": recent_avg * 1000,
                "pct": (total / total_all * 100) if total_all > 0 else 0,
                "calls": count,
            }

        return result

    def report_str(self) -> str:
        """Human-readable timing report."""
        stats = self.report()
        if not stats:
            return "  Profiler: no data"

        lines = [
            "  Performance Profile:",
            f"    {'Component':<20s} {'Avg ms':>8s} {'Recent':>8s} {'% Total':>8s} {'Calls':>8s}",
            f"    {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}",
        ]

        for comp, s in stats.items():
            lines.append(
                f"    {comp:<20s} {s['avg_ms']:>7.2f}ms {s['recent_avg_ms']:>7.2f}ms"
                f" {s['pct']:>7.1f}% {s['calls']:>7,}"
            )

        total_ms = self._step_total * 1000
        if self._step_count > 0:
            lines.append(f"    {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
            lines.append(
                f"    {'TOTAL':<20s} {total_ms / self._step_count:>7.2f}ms"
                f" {'':>8s} {'100.0':>7s}% {self._step_count:>7,}"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all accumulators."""
        self._totals.clear()
        self._counts.clear()
        self._recent.clear()
        self._current_start.clear()
        self._step_total = 0.0
        self._step_count = 0
        self._step_start = 0.0
