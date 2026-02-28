"""
auto_metric.py — Self-generating measurement discovery for novel environments.

The AutoMetric system solves the problem: "We don't know which aspects of
the raw state are predictive of good outcomes."

It works by:
  1. Recording raw state snapshots during episodes (no domain knowledge needed)
  2. Computing many candidate scalar reductions of those snapshots
  3. Correlating each candidate with episode outcome (lines, score, survival)
  4. Promoting high-correlation candidates to named hypotheses in the DB
  5. Optionally asking an LLM to name, describe, and extend the discoveries

The system is environment-agnostic: it works on any env that returns a
numpy-array observation. All you need is:
    - A raw state array at each episode end
    - A scalar outcome value (lines cleared, score, etc.)

Canonical usage
---------------
    # In FastLoop / episode runner:
    am = AutoMetric(db, game='tetris')
    am.record(raw_state=board_flat, outcome=lines_cleared, episode_id=eid)

    # In SlowLoop (after collecting N episodes):
    discoveries = am.analyze(min_episodes=50)
    for d in discoveries:
        print(d)   # e.g. "col_max_7: r=0.71 → likely max column height"

Design principles
-----------------
- Fully online: recording is O(1) per episode
- Agnostic: works on any fixed-shape observation
- Conservative: only promote features with strong evidence (r > threshold,
  N > min_episodes)
- Transparent: every discovery is logged with its correlation and sample size
- Extensible: LLM callback receives raw discoveries and can add semantic labels
"""

from __future__ import annotations

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from collections import defaultdict


# ── Candidate scalar extractors ────────────────────────────────────────────

def _extract_candidates(obs: np.ndarray, prefix: str = '') -> Dict[str, float]:
    """
    Compute all candidate scalar measurements from a raw observation array.

    Works on ANY shape — flattened, 1D, 2D (treated as matrix).
    Returns a flat dict of {name: float_value}.

    For a 1D array of length N (e.g. a flat board):
        - raw_{i}:       raw value at position i  (only first 32 tracked)
        - mean, std, max, min, sum
        - pct_filled:    fraction > 0
        - top_half_mean, bottom_half_mean, left_half_mean, right_half_mean
        - entropy:       -sum(p*log(p)) of histogram

    For a 2D array (H x W):
        - col_max_{c}:   max value in column c
        - col_mean_{c}:  mean value in column c
        - col_height_{c}: first nonzero row index (from top)
        - row_density_{r}: fraction of filled cells in row r (top 8 rows)
        - All the 1D reductions of the flattened array
    """
    obs = np.asarray(obs, dtype=np.float32).ravel()  # always work on flat
    n = len(obs)
    result = {}
    p = prefix

    # Global statistics
    result[f'{p}mean']      = float(np.mean(obs))
    result[f'{p}std']       = float(np.std(obs))
    result[f'{p}max_val']   = float(np.max(obs))
    result[f'{p}min_val']   = float(np.min(obs))
    result[f'{p}sum']       = float(np.sum(obs))
    result[f'{p}pct_filled'] = float(np.mean(obs > 0))

    # Half-splits (spatial structure)
    half = n // 2
    if half > 0:
        result[f'{p}top_half_mean']    = float(np.mean(obs[:half]))
        result[f'{p}bottom_half_mean'] = float(np.mean(obs[half:]))
        quarter = n // 4
        if quarter > 0:
            result[f'{p}left_half_mean']  = float(np.mean(obs[::2]))   # even indices
            result[f'{p}right_half_mean'] = float(np.mean(obs[1::2]))  # odd indices
            result[f'{p}top_quarter']     = float(np.mean(obs[:quarter]))
            result[f'{p}bottom_quarter']  = float(np.mean(obs[-quarter:]))

    # Entropy (how mixed/ordered is the distribution)
    hist, _ = np.histogram(obs, bins=8, range=(0, 1))
    hist = hist + 1e-8
    hist = hist / hist.sum()
    result[f'{p}entropy'] = float(-np.sum(hist * np.log(hist + 1e-8)))

    # Runs of consecutive filled cells (proxy for "flatness")
    binary = (obs > 0).astype(int)
    diffs = np.diff(binary)
    result[f'{p}n_transitions'] = float(np.sum(np.abs(diffs)))

    # Max run length of filled cells
    max_run = 0
    cur_run = 0
    for b in binary:
        if b:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    result[f'{p}max_run'] = float(max_run)

    # Individual positions (only first 32 — beyond that it's noise)
    for i in range(min(32, n)):
        result[f'{p}pos_{i}'] = float(obs[i])

    return result


def _extract_2d_candidates(obs_2d: np.ndarray,
                           prefix: str = '') -> Dict[str, float]:
    """
    Additional candidates for 2D board-like observations (H x W).
    """
    board = np.asarray(obs_2d, dtype=np.float32)
    if board.ndim != 2:
        return {}

    H, W = board.shape
    result = {}
    p = prefix

    # Per-column features (most useful for stack-based games like Tetris)
    col_heights = []
    for c in range(W):
        col = board[:, c]
        filled_rows = np.where(col > 0)[0]
        h = H - filled_rows[0] if len(filled_rows) > 0 else 0
        result[f'{p}col_height_{c}'] = float(h)
        result[f'{p}col_mean_{c}']   = float(np.mean(col))
        result[f'{p}col_max_{c}']    = float(np.max(col))
        col_heights.append(h)

    if col_heights:
        result[f'{p}col_height_std']   = float(np.std(col_heights))
        result[f'{p}col_height_max']   = float(max(col_heights))
        result[f'{p}col_height_min']   = float(min(col_heights))
        result[f'{p}col_height_range'] = float(max(col_heights) - min(col_heights))
        # Bumpiness (automatic!)
        result[f'{p}bumpiness'] = float(
            sum(abs(col_heights[i] - col_heights[i+1]) for i in range(len(col_heights)-1))
        )

    # Per-row features (top 8 rows — most informative for death prediction)
    for r in range(min(8, H)):
        result[f'{p}row_density_{r}'] = float(np.mean(board[r, :] > 0))

    # Holes (automatic!)
    holes = 0
    for c in range(W):
        found = False
        for r in range(H):
            if board[r, c] > 0:
                found = True
            elif found:
                holes += 1
    result[f'{p}holes'] = float(holes)

    # Aggregate height
    result[f'{p}agg_height'] = float(sum(col_heights))

    # Top-heaviness (more filled in top half = danger)
    result[f'{p}top_fill'] = float(np.mean(board[:H//2, :] > 0))
    result[f'{p}bot_fill'] = float(np.mean(board[H//2:, :] > 0))
    result[f'{p}top_heaviness'] = result[f'{p}top_fill'] - result[f'{p}bot_fill']

    return result


# ── AutoMetric ─────────────────────────────────────────────────────────────

class MetricDiscovery:
    """Single discovered metric entry."""
    __slots__ = ['name', 'correlation', 'n_samples', 'mean_val', 'std_val',
                 'description', 'formula', 'created_at']

    def __init__(self, name: str, correlation: float, n_samples: int,
                 mean_val: float, std_val: float,
                 description: str = '', formula: str = ''):
        self.name        = name
        self.correlation = correlation
        self.n_samples   = n_samples
        self.mean_val    = mean_val
        self.std_val     = std_val
        self.description = description
        self.formula     = formula
        self.created_at  = time.time()

    def __repr__(self):
        sign = '+' if self.correlation > 0 else ''
        return (f"MetricDiscovery({self.name!r}: r={sign}{self.correlation:.3f} "
                f"n={self.n_samples} mean={self.mean_val:.2f})")


class AutoMetric:
    """
    Self-generating measurement discovery for novel environments.

    Records raw state observations + outcomes, then periodically runs
    correlation analysis to discover which state features best predict
    good vs bad outcomes.

    No domain knowledge required. Works on any environment.
    """

    def __init__(self,
                 db=None,
                 game: str = 'unknown',
                 obs_shape: Optional[Tuple] = None,
                 min_correlation: float = 0.40,
                 min_episodes: int = 50,
                 max_candidates: int = 200,
                 llm_callback: Optional[Callable] = None,
                 storage_path: str = 'experiments/auto_metrics.jsonl'):
        """
        Args:
            db:               ExperimentDB for writing discovered hypotheses.
            game:             Game tag.
            obs_shape:        If 2D, enables column/row analysis.
                              If None, inferred from first observation.
            min_correlation:  Minimum |r| to promote a feature.
            min_episodes:     Minimum samples before running analysis.
            max_candidates:   Max scalar candidates to track per episode.
            llm_callback:     Optional fn(discoveries) → dict of {name: description}.
                              Called to get human-readable labels for discovered features.
            storage_path:     JSONL file for recording observations.
        """
        self.db              = db
        self.game            = game
        self.obs_shape       = obs_shape
        self.min_correlation = min_correlation
        self.min_episodes    = min_episodes
        self.max_candidates  = max_candidates
        self.llm_callback    = llm_callback
        self.storage_path    = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory buffer: list of {candidates: dict, outcome: float}
        self._buffer: List[Dict] = []

        # Previously discovered metrics (to avoid re-discovering)
        self._known: set = set()

        # Candidate accumulator: {name: ([values], [outcomes])}
        self._cand_vals: Dict[str, List[float]] = defaultdict(list)
        self._outcomes:  List[float] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def record(self,
               raw_obs: np.ndarray,
               outcome: float,
               episode_id: Optional[str] = None,
               extra: Optional[Dict] = None) -> None:
        """
        Record a raw observation + outcome for one episode.

        Args:
            raw_obs:    Raw numpy array from the environment (board, pixels, etc.)
            outcome:    Scalar outcome (lines_cleared, score, survival_steps, etc.)
            episode_id: Optional episode ID for traceability.
            extra:      Optional extra scalars to also correlate (already extracted).
        """
        obs = np.asarray(raw_obs, dtype=np.float32)

        # Extract all candidates
        candidates = _extract_candidates(obs.ravel())

        # If 2D, also extract 2D-specific features
        if obs.ndim == 2 or (self.obs_shape and len(self.obs_shape) == 2):
            if obs.ndim == 1 and self.obs_shape:
                obs_2d = obs.reshape(self.obs_shape)
            else:
                obs_2d = obs
            candidates.update(_extract_2d_candidates(obs_2d))

        # Merge any pre-extracted extras
        if extra:
            candidates.update(extra)

        # Cap to max_candidates (drop least useful beyond limit)
        if len(candidates) > self.max_candidates:
            keys = list(candidates.keys())[:self.max_candidates]
            candidates = {k: candidates[k] for k in keys}

        # Accumulate into correlation buffers
        self._outcomes.append(float(outcome))
        for name, val in candidates.items():
            self._cand_vals[name].append(float(val))

        # Also persist to disk (lightweight JSONL)
        record = {'outcome': outcome, 'ts': time.time()}
        if episode_id:
            record['episode_id'] = episode_id
        if extra:
            record.update(extra)
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def analyze(self, min_episodes: Optional[int] = None) -> List[MetricDiscovery]:
        """
        Run correlation analysis on accumulated observations.

        Computes Pearson r between each candidate feature and outcome.
        Promotes high-correlation features as hypothesis candidates in the DB.

        Returns:
            List of MetricDiscovery objects, sorted by |correlation|.
        """
        min_ep = min_episodes or self.min_episodes
        n = len(self._outcomes)

        if n < min_ep:
            print(f"[AutoMetric] Only {n} episodes — need {min_ep} to analyze.")
            return []

        outcomes = np.array(self._outcomes)
        oy_std = np.std(outcomes)
        if oy_std < 1e-8:
            print("[AutoMetric] Outcome has zero variance — nothing to correlate against.")
            return []

        # Compute Pearson r for each candidate
        discoveries = []
        for name, vals in self._cand_vals.items():
            if len(vals) < min_ep:
                continue

            x = np.array(vals[-n:])   # align with outcomes
            if len(x) != n:
                continue

            x_std = np.std(x)
            if x_std < 1e-8:
                continue  # constant feature — useless

            # Pearson correlation
            r = float(np.corrcoef(x, outcomes)[0, 1])
            if np.isnan(r):
                continue

            if abs(r) >= self.min_correlation and name not in self._known:
                discoveries.append(MetricDiscovery(
                    name=name,
                    correlation=round(r, 4),
                    n_samples=n,
                    mean_val=round(float(np.mean(x)), 3),
                    std_val=round(float(np.std(x)), 3),
                    description=self._auto_describe(name, r),
                ))

        # Sort by |r| descending
        discoveries.sort(key=lambda d: abs(d.correlation), reverse=True)

        # Optional LLM enrichment
        if self.llm_callback and discoveries:
            try:
                labels = self.llm_callback(discoveries)
                if isinstance(labels, dict):
                    for d in discoveries:
                        if d.name in labels:
                            d.description = labels[d.name]
            except Exception as e:
                print(f"[AutoMetric] LLM callback failed: {e}")

        # Write to DB as hypothesis candidates
        if self.db and discoveries:
            self._promote_to_db(discoveries)

        # Mark as known to avoid re-promoting on next analyze()
        for d in discoveries:
            self._known.add(d.name)

        return discoveries

    def summary(self) -> str:
        """Human-readable summary of current recording state."""
        n = len(self._outcomes)
        if not self._outcomes:
            return "[AutoMetric] No episodes recorded yet."
        out = np.array(self._outcomes)
        return (
            f"[AutoMetric] {n} episodes recorded  "
            f"outcome: mean={np.mean(out):.1f} std={np.std(out):.1f} "
            f"min={np.min(out):.0f} max={np.max(out):.0f}  "
            f"candidates: {len(self._cand_vals)}"
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _auto_describe(name: str, r: float) -> str:
        """Generate a placeholder description from feature name + correlation sign."""
        direction = "positively" if r > 0 else "negatively"
        strength  = "strongly" if abs(r) > 0.6 else "moderately"
        # Parse the name for semantic hints
        hints = {
            'holes':         'number of cells buried beneath a filled cell',
            'bumpiness':     'sum of absolute column height differences',
            'col_height':    'height of column',
            'max_run':       'longest consecutive run of filled cells',
            'top_fill':      'density of filled cells in the top half of the board',
            'top_heaviness': 'how much heavier the top half is vs bottom',
            'entropy':       'distribution entropy of cell values',
            'n_transitions': 'number of filled/empty transitions in the flat board',
            'pct_filled':    'fraction of board cells that are filled',
            'std':           'standard deviation of cell values',
            'mean':          'mean cell value (overall fill density)',
        }
        base = ''
        for key, desc in hints.items():
            if key in name:
                base = f' — {desc}'
                break
        return (f"Auto-discovered: {strength} {direction} correlated with outcome "
                f"(r={r:+.3f}){base}. Requires validation.")

    def _promote_to_db(self, discoveries: List[MetricDiscovery]) -> None:
        """Write discovered metrics as hypothesis candidates in ExperimentDB."""
        if not self.db:
            return

        for d in discoveries:
            try:
                self.db.upsert_hypothesis(
                    name=f'auto_{d.name}',
                    game=self.game,
                    description=d.description,
                    confidence=min(0.9, 0.5 + abs(d.correlation) * 0.5),
                    win_rate=0.0,
                    evidence_count=d.n_samples,
                    status='candidate',
                    metadata={
                        'source':         'auto_metric',
                        'correlation':    d.correlation,
                        'mean_val':       d.mean_val,
                        'std_val':        d.std_val,
                        'discovered_at':  time.strftime('%Y-%m-%d %H:%M'),
                        'positive_signal': d.correlation > 0,
                    },
                )
            except Exception as e:
                print(f"[AutoMetric] DB write failed for {d.name}: {e}")
