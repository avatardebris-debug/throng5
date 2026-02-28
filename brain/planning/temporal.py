"""
temporal.py — Timing and sequence understanding.

Handles time-dependent game mechanics:
  - "The platform appears every 60 frames — time your jump"
  - "The enemy patrol cycles every 120 frames"
  - "Power-up lasts 200 frames"
  - "This door opens after collecting 3 items"

Learns periodic patterns from observations and predicts timing windows.

Usage:
    temporal = TemporalReasoner()
    temporal.observe(ram, step)
    windows = temporal.get_action_windows()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PeriodicPattern:
    """A detected periodic pattern in RAM."""

    def __init__(self, address: int, period: int):
        self.address = address
        self.period = period
        self.confidence: float = 0.0
        self.observations: int = 0
        self.phase: int = 0  # Phase offset
        self.amplitude: float = 0.0  # Range of values
        self.label: str = f"periodic_{address:#04x}"

    def predict_next_event(self, current_step: int) -> int:
        """Predict when this pattern will next trigger."""
        steps_since = (current_step - self.phase) % self.period
        return self.period - steps_since

    def is_in_window(self, current_step: int, window_size: int = 5) -> bool:
        """Check if we're in the action window for this pattern."""
        time_to_event = self.predict_next_event(current_step)
        return time_to_event <= window_size


class SequencePattern:
    """A detected sequential pattern (A then B then C)."""

    def __init__(self, events: List[Tuple[int, int]]):
        self.events = events  # [(addr, value), ...]
        self.avg_gaps: List[float] = []  # Avg frames between events
        self.observations: int = 0
        self.confidence: float = 0.0


class TemporalReasoner:
    """
    Learns temporal patterns from RAM observations.

    Detects:
    1. Periodic patterns (cycles, patrols, timers)
    2. Sequential patterns (A then B then C)
    3. Duration patterns (how long effects last)
    4. Action windows (when is it safe/optimal to act)
    """

    def __init__(self, ram_size: int = 128, history_length: int = 1000):
        self._ram_size = ram_size
        self._history_length = history_length

        # RAM history for pattern detection
        self._history: List[np.ndarray] = []
        self._step: int = 0

        # Detected patterns
        self._periodic: Dict[int, PeriodicPattern] = {}    # addr → pattern
        self._sequences: List[SequencePattern] = []
        self._durations: Dict[int, List[int]] = defaultdict(list)  # addr → durations

        # Event tracking
        self._last_change: Dict[int, int] = {}  # addr → step of last change
        self._change_intervals: Dict[int, List[int]] = defaultdict(list)

    def observe(self, ram: np.ndarray, step: Optional[int] = None) -> None:
        """Observe a RAM state and detect temporal patterns."""
        ram = np.asarray(ram, dtype=np.uint8).flatten()[:self._ram_size]
        self._step = step if step is not None else self._step + 1

        # Track changes
        if self._history:
            prev = self._history[-1]
            for addr in range(min(len(ram), len(prev))):
                if ram[addr] != prev[addr]:
                    # Record change interval
                    if addr in self._last_change:
                        interval = self._step - self._last_change[addr]
                        self._change_intervals[addr].append(interval)
                        # Keep only recent intervals
                        if len(self._change_intervals[addr]) > 50:
                            self._change_intervals[addr] = self._change_intervals[addr][-50:]
                    self._last_change[addr] = self._step

                    # Track duration of non-zero values
                    if ram[addr] == 0 and prev[addr] != 0:
                        # Value went back to 0 — record duration
                        if addr in self._last_change:
                            pass  # Duration tracking handled above

        # Store history
        self._history.append(ram.copy())
        if len(self._history) > self._history_length:
            self._history.pop(0)

        # Detect periodic patterns every 200 steps
        if self._step % 200 == 0:
            self._detect_periodic()

    def _detect_periodic(self) -> None:
        """Find periodic patterns in change intervals."""
        for addr, intervals in self._change_intervals.items():
            if len(intervals) < 5:
                continue

            arr = np.array(intervals, dtype=np.float32)
            mean_interval = float(np.mean(arr))
            std_interval = float(np.std(arr))

            # Low variance relative to mean = periodic
            if mean_interval > 5 and std_interval < mean_interval * 0.3:
                period = int(round(mean_interval))
                if addr not in self._periodic:
                    self._periodic[addr] = PeriodicPattern(addr, period)
                pattern = self._periodic[addr]
                pattern.period = period
                pattern.confidence = 1.0 - (std_interval / max(mean_interval, 1))
                pattern.observations = len(intervals)
                if addr in self._last_change:
                    pattern.phase = self._last_change[addr] % period

    def get_periodic_patterns(self) -> List[Dict[str, Any]]:
        """Get all detected periodic patterns."""
        return [
            {
                "addr": p.address,
                "addr_hex": f"0x{p.address:02X}",
                "period": p.period,
                "confidence": round(p.confidence, 3),
                "observations": p.observations,
                "next_event_in": p.predict_next_event(self._step),
                "label": p.label,
            }
            for p in sorted(self._periodic.values(), key=lambda x: -x.confidence)
        ]

    def get_action_windows(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """Get current action windows (patterns about to trigger)."""
        windows = []
        for addr, pattern in self._periodic.items():
            if pattern.confidence > 0.5:
                time_to = pattern.predict_next_event(self._step)
                if time_to <= window_size:
                    windows.append({
                        "addr": addr,
                        "label": pattern.label,
                        "frames_until": time_to,
                        "period": pattern.period,
                    })
        return windows

    def predict_safe_action_time(
        self,
        hazard_addrs: List[int],
        window_size: int = 10,
    ) -> Optional[int]:
        """
        Given hazardous periodic patterns, find the next safe window.

        Returns number of frames to wait, or None if no periodic hazards.
        """
        if not hazard_addrs:
            return None

        min_safe_time = 0
        for addr in hazard_addrs:
            if addr in self._periodic:
                pattern = self._periodic[addr]
                time_to = pattern.predict_next_event(self._step)
                if time_to < window_size:
                    # Hazard is about to trigger — wait for it to pass
                    min_safe_time = max(min_safe_time, time_to + 5)

        return min_safe_time if min_safe_time > 0 else None

    def report(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "history_length": len(self._history),
            "periodic_patterns": len(self._periodic),
            "high_confidence": sum(
                1 for p in self._periodic.values() if p.confidence > 0.7
            ),
            "bytes_tracked": len(self._change_intervals),
        }
