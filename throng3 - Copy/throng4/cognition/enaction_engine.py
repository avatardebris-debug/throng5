"""
enaction_engine.py — Translate promoted hypothesis programs into live agent config.

An enacted hypothesis is a hypothesis with an `enaction` field in its metadata.
Supported types (Tetra's three-type vocabulary):

    reward_weight:
        {"type": "reward_weight", "target": "holes", "multiplier": 1.5}
        Scales a DellacherieWeights field. Valid targets: holes, aggregate_height,
        bumpiness, lines_cleared.

    mode_gate:
        {"type": "mode_gate", "condition": "height > 0.75", "strategy": "flat_over_lines"}
        Switches ModeController behavior based on board state. Supported conditions:
            height > <fraction>   (fraction of board height, 0.0–1.0)
            holes > <count>
            pieces < <count>      (early-game gate)
        Supported strategies:
            flat_over_lines       (prefer flat placements, deprioritise line clears)
            survive               (force SURVIVE mode regardless of threat score)

    piece_phase:
        {"type": "piece_phase", "range": [0, 10], "multiplier": 2.0, "target": "holes"}
        Applies a reward_weight multiplier only during pieces [range[0], range[1]].

The engine reads experiments/active_enactions.json (written by SlowLoop at promotion
time) and returns a merged config dict that FastLoop applies to each episode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


ENACTION_FILE = 'experiments/active_enactions.json'

# Valid reward_weight targets and their DellacherieWeights field names
WEIGHT_TARGETS = {
    'holes':            'holes',
    'aggregate_height': 'aggregate_height',
    'bumpiness':        'bumpiness',
    'lines_cleared':    'lines_cleared',
}


@dataclass
class EnactionConfig:
    """
    Merged live config derived from all active enacted hypotheses.

    Passed to TetrisAdapter/TetrisCurriculumEnv each episode.
    """
    # reward_weight multipliers (multiplicative on DellacherieWeights)
    holes_multiplier:    float = 1.0
    height_multiplier:   float = 1.0
    bumpiness_multiplier: float = 1.0
    lines_multiplier:    float = 1.0

    # mode_gate rules (applied by ModeController / FastLoop per-step)
    mode_gates: List[Dict[str, Any]] = None

    # piece_phase rules (applied per-step in _run_episode)
    piece_phases: List[Dict[str, Any]] = None

    # Source hypotheses that produced this config (for logging)
    sources: List[str] = None

    def __post_init__(self):
        if self.mode_gates   is None: self.mode_gates   = []
        if self.piece_phases is None: self.piece_phases = []
        if self.sources      is None: self.sources      = []

    def is_identity(self) -> bool:
        """True if this config makes no changes from the default."""
        return (
            self.holes_multiplier    == 1.0 and
            self.height_multiplier   == 1.0 and
            self.bumpiness_multiplier == 1.0 and
            self.lines_multiplier    == 1.0 and
            not self.mode_gates and
            not self.piece_phases
        )

    def apply_to_weights(self, weights) -> None:
        """
        Mutate a DellacherieWeights instance in-place with active multipliers.
        Called once per episode in FastLoop before creating TetrisAdapter.
        """
        weights.holes            *= self.holes_multiplier
        weights.aggregate_height *= self.height_multiplier
        weights.bumpiness        *= self.bumpiness_multiplier
        weights.lines_cleared    *= self.lines_multiplier

    def get_phase_multiplier(self, target: str, pieces_placed: int) -> float:
        """
        Return the combined multiplier for a reward target at a given piece count.
        Piece-phase rules are additive (stack multiplicatively).
        """
        m = 1.0
        for rule in self.piece_phases:
            lo, hi = rule.get('range', [0, 999])
            if lo <= pieces_placed <= hi and rule.get('target') == target:
                m *= rule.get('multiplier', 1.0)
        return m

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnactionEngine:
    """
    Reads active_enactions.json and builds a merged EnactionConfig.

    Usage in FastLoop:
        engine = EnactionEngine()
        config = engine.load()                    # call at pack reload
        weights = DellacherieWeights()
        config.apply_to_weights(weights)          # mutates weights for this episode
        adapter = TetrisAdapter(level=7, weights=weights, ...)
    """

    def __init__(self, path: str = ENACTION_FILE):
        self.path = Path(path)
        self._last_mtime: float = 0.0
        self._cached: EnactionConfig = EnactionConfig()

    def load(self, force: bool = False) -> EnactionConfig:
        """
        Load and merge all enactions from file. Caches result;
        re-reads only if file has changed (or force=True).
        """
        if not self.path.exists():
            return EnactionConfig()

        mtime = self.path.stat().st_mtime
        if not force and mtime == self._last_mtime:
            return self._cached

        try:
            enactions = json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            return self._cached

        self._last_mtime = mtime
        self._cached = self._merge(enactions)
        return self._cached

    @staticmethod
    def _merge(enactions: List[Dict[str, Any]]) -> EnactionConfig:
        """Merge a list of enaction dicts into one EnactionConfig."""
        config = EnactionConfig()

        for e in enactions:
            etype  = e.get('type', '').lower()
            source = e.get('hypothesis', 'unknown')

            if etype == 'reward_weight':
                target = e.get('target', '')
                mult   = float(e.get('multiplier', 1.0))
                if target == 'holes':
                    config.holes_multiplier *= mult
                elif target == 'aggregate_height':
                    config.height_multiplier *= mult
                elif target == 'bumpiness':
                    config.bumpiness_multiplier *= mult
                elif target == 'lines_cleared':
                    config.lines_multiplier *= mult
                config.sources.append(source)

            elif etype == 'mode_gate':
                config.mode_gates.append({
                    'condition': e.get('condition', ''),
                    'strategy':  e.get('strategy', 'survive'),
                    'source':    source,
                })
                config.sources.append(source)

            elif etype == 'piece_phase':
                config.piece_phases.append({
                    'range':      e.get('range', [0, 10]),
                    'target':     e.get('target', 'holes'),
                    'multiplier': float(e.get('multiplier', 1.0)),
                    'source':     source,
                })
                config.sources.append(source)

        return config

    @staticmethod
    def write_from_hypotheses(hypotheses: List[Dict[str, Any]],
                               path: str = ENACTION_FILE) -> int:
        """
        Extract enaction schemas from promoted hypothesis metadata and write
        active_enactions.json. Called by SlowLoop at promotion time.

        Returns number of enacted hypotheses written.
        """
        enactions: List[Dict[str, Any]] = []
        for h in hypotheses:
            meta = h.get('metadata') or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue
            enaction = meta.get('enaction')
            if enaction and isinstance(enaction, dict):
                enaction['hypothesis'] = h.get('name', 'unknown')
                enactions.append(enaction)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(enactions, indent=2), encoding='utf-8')
        return len(enactions)
