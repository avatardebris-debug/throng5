"""
telemetry_logger.py — Append-only JSONL episode trace logger.

Complements ExperimentDB (structured queries) with a flat, greppable,
human-readable log of every episode. One file per calendar day.

Design
------
- Zero overhead: open → write one JSON line → close. No transactions.
- Date-rotated: experiments/logs/YYYY-MM-DD.jsonl
- read_since(ts): returns all records after a Unix timestamp — used by
  the slow loop to find unprocessed episodes since last consolidation.
- Compact record: only the fields needed for pattern analysis. Full
  board state is NOT stored here (that's the DB's job).

Usage
-----
    logger = TelemetryLogger()

    # After each episode:
    logger.log_episode({
        'game': 'tetris', 'level': 2, 'episode': 42,
        'lines': 7, 'score': 88.0, 'pieces': 34,
        'fail_mode': 'timing', 'best_hypothesis': 'maximize_gain_v2',
        'dreamer_nudges': 3, 'exec_nudges': 1,
    })

    # Slow loop reads unprocessed episodes:
    new_eps = logger.read_since(last_consolidation_ts)
"""

from __future__ import annotations

import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


class TelemetryLogger:
    """
    Append-only JSONL logger for per-episode telemetry.

    One file per calendar day under log_dir/YYYY-MM-DD.jsonl.
    Each line is a self-contained JSON record with a 'timestamp' field.

    Thread-safety: not thread-safe — designed for single-process use.
    For multi-process, use separate log_dirs per process and merge offline.
    """

    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_episode(self, data: Dict[str, Any],
                    timestamp: Optional[float] = None) -> None:
        """
        Append one episode record to today's JSONL file.

        Args:
            data:      Episode data dict. Any keys are accepted; 'timestamp'
                       is added automatically if not present.
            timestamp: Unix timestamp override (default: now).
        """
        record = dict(data)
        record.setdefault('timestamp', timestamp or time.time())

        log_file = self._today_file()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def log_event(self, event_type: str, data: Optional[Dict] = None,
                  timestamp: Optional[float] = None) -> None:
        """
        Append a non-episode event (novelty flag, hypothesis mutation, etc).

        Args:
            event_type: e.g. 'novelty', 'mutation', 'level_advance'
            data:       Optional payload dict.
        """
        record = {
            'event_type': event_type,
            'timestamp': timestamp or time.time(),
        }
        if data:
            record.update(data)
        log_file = self._today_file()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_since(self, since_timestamp: float,
                   game: Optional[str] = None,
                   level: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return all records with timestamp >= since_timestamp.

        Scans log files from oldest to newest, stopping once all files
        are exhausted. Only reads files that could contain records after
        since_timestamp (based on file date).

        Args:
            since_timestamp: Unix timestamp lower bound (inclusive).
            game:            Optional filter by game name.
            level:           Optional filter by level.

        Returns:
            List of record dicts, sorted by timestamp ascending.
        """
        results = []
        for log_file in sorted(self._all_log_files()):
            # Skip files that are definitely too old
            # File date is the day it was created — records in it could span
            # up to midnight of that day. We use a 1-day buffer to be safe.
            file_date_ts = self._file_date_ts(log_file)
            if file_date_ts + 86400 < since_timestamp:
                continue

            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ts = record.get('timestamp', 0)
                        if ts < since_timestamp:
                            continue
                        if game and record.get('game') != game:
                            continue
                        if level is not None and record.get('level') != level:
                            continue
                        results.append(record)
            except OSError:
                continue

        results.sort(key=lambda r: r.get('timestamp', 0))
        return results

    def read_today(self, game: Optional[str] = None,
                   level: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all records from today's log file."""
        midnight = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        return self.read_since(midnight, game=game, level=level)

    def tail(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last N records across all log files."""
        all_records = self.read_since(0)
        return all_records[-n:]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Quick summary: total records, date range, files on disk."""
        files = sorted(self._all_log_files())
        if not files:
            return {'total_records': 0, 'files': 0, 'oldest': None, 'newest': None}

        total = 0
        oldest_ts = float('inf')
        newest_ts = 0.0

        for log_file in files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                            ts = r.get('timestamp', 0)
                            oldest_ts = min(oldest_ts, ts)
                            newest_ts = max(newest_ts, ts)
                            total += 1
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

        return {
            'total_records': total,
            'files': len(files),
            'oldest': datetime.fromtimestamp(oldest_ts).isoformat() if oldest_ts != float('inf') else None,
            'newest': datetime.fromtimestamp(newest_ts).isoformat() if newest_ts else None,
            'log_dir': str(self.log_dir),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _today_file(self) -> Path:
        """Path to today's log file."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        return self.log_dir / f"{date_str}.jsonl"

    def _all_log_files(self) -> List[Path]:
        """All .jsonl files in log_dir, sorted by name (= date order)."""
        return sorted(self.log_dir.glob('*.jsonl'))

    @staticmethod
    def _file_date_ts(log_file: Path) -> float:
        """Parse the date from a YYYY-MM-DD.jsonl filename → Unix timestamp."""
        try:
            date_str = log_file.stem  # '2026-02-17'
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.timestamp()
        except ValueError:
            return 0.0
