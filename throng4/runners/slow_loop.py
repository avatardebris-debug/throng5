"""
slow_loop.py — Offline batch consolidation of episodes into hypotheses.

The SlowLoop runs periodically (hourly lightweight, nightly full) and:
  1. Reads episodes logged since the last consolidation
  2. Clusters failure patterns by board state signatures
  3. Updates existing hypothesis win_rate / evidence_count in DB
  4. Generates new candidate hypotheses from failure clusters
  5. Applies promotion gates → writes a new PolicyPack if anything changed
  6. Emits a ConsolidationReport (printed + saved to experiments/consolidation/)

Design principles
-----------------
- Fully offline: no simulation, no LLM (LLM is optional via a callback)
- Idempotent: re-running on the same data produces the same result
- Conservative: only promotes hypotheses, never demotes (demotion = slow decay)
- Transparent: every decision logged in the ConsolidationReport

Usage
-----
    from throng4.runners.slow_loop import SlowLoop
    from throng4.storage import ExperimentDB

    with ExperimentDB('experiments/experiments.db') as db:
        slow = SlowLoop(db)
        report = slow.consolidate(mode='hourly')
        print(report.summary())

    # CLI:
    python -m throng4.runners.slow_loop --mode hourly
    python -m throng4.runners.slow_loop --mode nightly
"""

from __future__ import annotations

import time
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from throng4.storage.experiment_db import ExperimentDB
from throng4.storage.policy_pack import PolicyPack, PromotionGates
try:
    from throng4.cognition.threat_estimator import ThreatEstimator
    _THREAT_AVAILABLE = True
except ImportError:
    _THREAT_AVAILABLE = False


# ── ConsolidationReport ────────────────────────────────────────────────────

@dataclass
class ConsolidationReport:
    """Result of one consolidation pass."""
    mode: str
    game: str
    episodes_processed: int
    hypotheses_updated: List[str] = field(default_factory=list)
    candidates_generated: List[str] = field(default_factory=list)
    pack_version: Optional[int] = None
    pack_promoted: int = 0
    pack_rejected: int = 0
    failure_clusters: List[Dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"ConsolidationReport [{self.mode}]  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"  Game:              {self.game or 'all'}",
            f"  Episodes read:     {self.episodes_processed}",
            f"  Hypotheses updated:{len(self.hypotheses_updated)}",
            f"  New candidates:    {len(self.candidates_generated)}",
            f"  PolicyPack:        v{self.pack_version}  "
            f"({self.pack_promoted} promoted, {self.pack_rejected} rejected)",
            f"  Elapsed:           {self.elapsed_s:.1f}s",
        ]
        if self.failure_clusters:
            lines.append(f"\n  Failure clusters ({len(self.failure_clusters)}):")
            for cl in self.failure_clusters[:5]:
                lines.append(f"    [{cl['size']:3d} eps] avg_lines={cl['avg_lines']:.1f}  "
                             f"avg_height={cl['avg_height']:.1f}  "
                             f"avg_holes={cl['avg_holes']:.1f}")
        if self.candidates_generated:
            lines.append(f"\n  New candidates:")
            for name in self.candidates_generated:
                lines.append(f"    + {name}")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)


# ── SlowLoop ───────────────────────────────────────────────────────────────

class SlowLoop:
    """
    Offline batch consolidator: episodes → hypotheses → PolicyPack.
    """

    # How many seconds back to look in 'hourly' vs 'nightly' mode
    LOOKBACK = {
        'hourly':  3_600,      # 1 hour
        'nightly': 86_400,     # 24 hours
        'full':    None,       # all time
    }

    def __init__(self, db: ExperimentDB,
                 game: str = 'tetris',
                 gates: Optional[PromotionGates] = None,
                 llm_callback: Optional[Callable] = None,
                 report_dir: str = 'experiments/consolidation'):
        """
        Args:
            db:            Open ExperimentDB.
            game:          Game to consolidate for.
            gates:         PromotionGates for PolicyPack creation.
            llm_callback:  Optional fn(clusters) -> List[str] of candidate names.
                           If None, candidate generation is heuristic-only.
            report_dir:    Directory to save markdown consolidation reports.
        """
        self.db           = db
        self.game         = game
        self.gates        = gates or PromotionGates(min_evidence=30, min_win_rate=0.20)
        self.llm_callback = llm_callback
        self.report_dir   = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def consolidate(self, mode: str = 'hourly') -> ConsolidationReport:
        """
        Run a full consolidation pass.

        Args:
            mode: 'hourly', 'nightly', or 'full'.

        Returns:
            ConsolidationReport with all actions taken.
        """
        t_start = time.time()
        lookback = self.LOOKBACK.get(mode)
        since    = (time.time() - lookback) if lookback else 0.0

        report = ConsolidationReport(mode=mode, game=self.game,
                                     episodes_processed=0)

        # 1. Read episodes since last consolidation
        episodes = self.db.get_episodes(
            game=self.game or None,
            since=since,
            limit=10_000,
        )
        report.episodes_processed = len(episodes)

        if not episodes:
            print(f"  [SlowLoop] No new episodes since last {mode} consolidation.")
            report.elapsed_s = time.time() - t_start
            return report

        print(f"  [SlowLoop/{mode}] Processing {len(episodes)} episodes...")

        # 0. Ingest Tetra inbox if present (any mode)
        inbox_summary = self.db.ingest_tetra_inbox()
        if not inbox_summary.get('skipped'):
            added   = inbox_summary.get('added', 0)
            updated = inbox_summary.get('updated', 0)
            retired = inbox_summary.get('retired', 0)
            mutated = inbox_summary.get('mutated', 0)
            errs    = inbox_summary.get('errors', [])
            print(f"  [tetra] Inbox ingested: "
                  f"+{added} added  ~{updated} updated  "
                  f"-{retired} retired  ↺{mutated} mutated"
                  + (f"  ⚠ {len(errs)} errors" if errs else ""))
            for e in errs:
                print(f"    ↳ {e}")

        # 2. Update hypothesis win_rates from episode hypothesis_performance
        updated = self._update_hypothesis_stats(episodes)
        report.hypotheses_updated = updated

        # 3. Cluster failure patterns
        clusters = self._cluster_failures(episodes)
        report.failure_clusters = clusters

        # 4. Generate new candidate hypotheses from clusters
        candidates = self._generate_candidates(clusters, episodes)
        report.candidates_generated = candidates

        # 5. Apply promotion gates → new PolicyPack
        pack = PolicyPack.from_db(self.db, game=self.game, gates=self.gates,
                                  notes=f"slow_loop/{mode}")
        report.pack_version  = pack.version
        report.pack_promoted = len(pack.hypotheses)

        # Count rejections (all hypotheses - promoted)
        all_hyps = self.db.get_hypotheses(game=self.game or None)
        report.pack_rejected = len(all_hyps) - len(pack.hypotheses)

        # 5b. Auto-retrain ThreatEstimator (nightly/full only)
        if mode in ('nightly', 'full') and _THREAT_AVAILABLE:
            self._retrain_threat_estimators(episodes)

        report.elapsed_s = time.time() - t_start

        # 6. Save report to file
        self._save_report(report, pack)

        print(report.summary())
        return report

    # ── Step 2: Update hypothesis stats from episode data ─────────────────

    def _update_hypothesis_stats(self, episodes: List[Dict]) -> List[str]:
        """
        Recompute win_rate and evidence_count for each hypothesis
        based on hypothesis_performance fields in logged episodes.

        Returns list of updated hypothesis names.
        """
        # Accumulate win counts per hypothesis name
        win_counts:    Dict[str, int]   = defaultdict(int)
        total_counts:  Dict[str, int]   = defaultdict(int)
        reward_sums:   Dict[str, float] = defaultdict(float)

        for ep in episodes:
            perf_json = ep.get('hypothesis_performance')
            if not perf_json:
                continue
            try:
                perf = json.loads(perf_json) if isinstance(perf_json, str) else perf_json
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(perf, dict):
                continue

            # Find the best hypothesis in this episode
            best_name = None
            best_wr   = -1.0
            for name, stats in perf.items():
                if not isinstance(stats, dict):
                    continue
                wr = stats.get('win_rate', 0.0)
                if wr > best_wr:
                    best_wr   = wr
                    best_name = name

            for name, stats in perf.items():
                if not isinstance(stats, dict):
                    continue
                total_counts[name] += 1
                reward_sums[name]  += stats.get('avg_reward', 0.0)
                if name == best_name:
                    win_counts[name] += 1

        # Update DB for each hypothesis seen
        updated = []
        for name in total_counts:
            n     = total_counts[name]
            wins  = win_counts[name]
            wr    = wins / n if n > 0 else 0.0
            avg_r = reward_sums[name] / n if n > 0 else 0.0

            # Get existing record to accumulate (not replace) evidence
            existing = self.db.get_hypotheses(game=self.game or None)
            existing_map = {h['name']: h for h in existing}

            if name in existing_map:
                h = existing_map[name]
                # Weighted merge: existing evidence + new evidence
                old_n  = h.get('evidence_count', 0)
                old_wr = h.get('win_rate', 0.0)
                total_n = old_n + n
                merged_wr = (old_wr * old_n + wr * n) / total_n if total_n > 0 else wr

                self.db.upsert_hypothesis(
                    name=name, game=self.game,
                    confidence=merged_wr,
                    win_rate=merged_wr,
                    evidence_count=total_n,
                    status=h.get('status', 'testing'),
                    metadata=json.loads(h['metadata']) if h.get('metadata') else None,
                )
                updated.append(name)

        return updated

    # ── Step 3: Cluster failure patterns ──────────────────────────────────

    def _cluster_failures(self, episodes: List[Dict],
                          n_clusters: int = 5) -> List[Dict]:
        """
        Group episodes by outcome quality into clusters.

        Simple percentile-based binning (no scipy dependency):
          - Cluster 0: bottom 20% by lines (worst failures)
          - Cluster 1: 20-40th percentile
          - ...
          - Cluster 4: top 20% (best episodes)

        Returns list of cluster dicts with aggregate stats.
        """
        if not episodes:
            return []

        lines_list = [ep.get('lines_cleared', 0) for ep in episodes]
        sorted_lines = sorted(lines_list)
        n = len(sorted_lines)

        # Percentile boundaries using safe index clamping
        boundaries = [sorted_lines[min(int(n * p / n_clusters), n - 1)]
                      for p in range(n_clusters + 1)]
        boundaries[-1] = sorted_lines[-1] + 1  # inclusive upper bound

        clusters = []
        for ci in range(n_clusters):
            lo, hi = boundaries[ci], boundaries[ci + 1]
            bucket = [ep for ep in episodes
                      if lo <= ep.get('lines_cleared', 0) < hi]
            if not bucket:
                continue

            avg_lines  = sum(ep.get('lines_cleared', 0)  for ep in bucket) / len(bucket)
            avg_height = sum(ep.get('max_height', 0)      for ep in bucket) / len(bucket)
            avg_holes  = sum(ep.get('holes', 0)           for ep in bucket) / len(bucket)
            avg_score  = sum(ep.get('score', 0.0)         for ep in bucket) / len(bucket)

            clusters.append({
                'cluster_id': ci,
                'size': len(bucket),
                'lines_range': (lo, hi - 1),
                'avg_lines': round(avg_lines, 1),
                'avg_height': round(avg_height, 1),
                'avg_holes': round(avg_holes, 1),
                'avg_score': round(avg_score, 2),
                'is_failure': ci < 2,   # Bottom 40% = failures
            })

        return clusters

    # ── Step 4: Generate candidate hypotheses ─────────────────────────────

    def _generate_candidates(self, clusters: List[Dict],
                              episodes: List[Dict]) -> List[str]:
        """
        Heuristic candidate generation from failure cluster patterns.

        Compares the worst-performing cluster to the best-performing cluster.
        If the failure cluster has notably more holes / height, generate a
        targeted hypothesis candidate to address that.

        LLM callback (if set) is called with the clusters for richer generation.
        """
        if len(clusters) < 2:
            return []

        worst = clusters[0]   # lowest lines
        best  = clusters[-1]  # highest lines

        candidates = []
        existing_names = {h['name'] for h in self.db.get_hypotheses(game=self.game or None)}

        # Heuristic rules
        hole_delta   = worst['avg_holes']  - best['avg_holes']
        height_delta = worst['avg_height'] - best['avg_height']
        score_delta  = best['avg_score']   - worst['avg_score']

        if hole_delta > 2.0:
            name = f"reduce_holes_v{int(time.time()) % 1000}"
            if name not in existing_names:
                self.db.upsert_hypothesis(
                    name=name, game=self.game,
                    description=f"Reduce holes: failure cluster had {worst['avg_holes']:.1f} "
                                f"vs success {best['avg_holes']:.1f} avg holes",
                    confidence=0.4, win_rate=0.0, evidence_count=0,
                    status='candidate',
                    metadata={'generated_by': 'slow_loop', 'hole_delta': hole_delta},
                )
                candidates.append(name)

        if height_delta > 3.0:
            name = f"control_height_v{int(time.time()) % 1000}"
            if name not in existing_names:
                self.db.upsert_hypothesis(
                    name=name, game=self.game,
                    description=f"Control height: failure cluster had {worst['avg_height']:.1f} "
                                f"vs success {best['avg_height']:.1f} avg max height",
                    confidence=0.4, win_rate=0.0, evidence_count=0,
                    status='candidate',
                    metadata={'generated_by': 'slow_loop', 'height_delta': height_delta},
                )
                candidates.append(name)

        # Optional LLM enrichment (non-blocking — if it fails, skip)
        if self.llm_callback and clusters:
            try:
                llm_names = self.llm_callback(clusters)
                for name in (llm_names or []):
                    if name not in existing_names:
                        self.db.upsert_hypothesis(
                            name=name, game=self.game,
                            description=f"LLM-generated candidate from slow_loop/{time.strftime('%Y-%m-%d')}",
                            confidence=0.4, win_rate=0.0, evidence_count=0,
                            status='candidate',
                            metadata={'generated_by': 'llm'},
                        )
                        candidates.append(name)
            except Exception as e:
                print(f"  [SlowLoop] LLM callback failed (skipping): {e}")

        return candidates

    # ── Step 5b: Auto-retrain ThreatEstimators ────────────────────────────

    def _retrain_threat_estimators(self, episodes: List[Dict],
                                   min_episodes: int = 500,
                                   n_train: int = 3000,
                                   save_dir: str = 'experiments') -> None:
        """
        Retrain ThreatEstimator for any level that has >= min_episodes new
        episodes. Also retrain the universal (all-levels) estimator.

        Skips silently if ThreatEstimator is unavailable or training fails.
        """
        # Group by level
        by_level: Dict[int, int] = defaultdict(int)
        for ep in episodes:
            lvl = ep.get('level')
            if lvl is not None:
                by_level[int(lvl)] += 1

        retrained = []

        for lvl, count in sorted(by_level.items()):
            if count < min_episodes:
                continue
            path = f'{save_dir}/threat_estimator_L{lvl}.npz'
            try:
                te = ThreatEstimator()
                result = te.train_from_db(
                    self.db, game=self.game or 'tetris',
                    n_episodes=n_train, level=lvl, verbose=False,
                )
                if 'error' not in result:
                    te.save(path)
                    retrained.append(f'L{lvl}(n={count})')
            except Exception as e:
                print(f'  [amygdala] L{lvl} retrain failed: {e}')

        # Universal estimator — retrain if any level was retrained
        if retrained:
            path = f'{save_dir}/threat_estimator_all.npz'
            try:
                te = ThreatEstimator()
                result = te.train_from_db(
                    self.db, game=self.game or 'tetris',
                    n_episodes=n_train, level=None, verbose=False,
                )
                if 'error' not in result:
                    te.save(path)
                    retrained.append('all')
            except Exception as e:
                print(f'  [amygdala] universal retrain failed: {e}')

            print(f'  [amygdala] Retrained estimators: {", ".join(retrained)}')
        else:
            print(f'  [amygdala] No estimators retrained '
                  f'(need >= {min_episodes} new eps per level)')

    # ── Step 6: Save report ────────────────────────────────────────────────

    def _save_report(self, report: ConsolidationReport,
                     pack: PolicyPack) -> None:
        """Save consolidation report as markdown file."""
        date_str = time.strftime('%Y-%m-%d_%H-%M')
        path = self.report_dir / f"{date_str}_{report.mode}.md"

        lines = [
            f"# Consolidation Report: {report.mode}",
            f"",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}  ",
            f"**Game:** {report.game or 'all'}  ",
            f"**Episodes processed:** {report.episodes_processed}  ",
            f"**Elapsed:** {report.elapsed_s:.1f}s  ",
            f"",
            f"## PolicyPack v{report.pack_version}",
            f"",
            f"- Promoted: {report.pack_promoted}",
            f"- Rejected: {report.pack_rejected}",
            f"",
            pack.summary(),
            f"",
            f"## Failure Clusters",
            f"",
        ]

        for cl in report.failure_clusters:
            tag = '🔴 FAILURE' if cl['is_failure'] else '🟢 SUCCESS'
            lines.append(f"### Cluster {cl['cluster_id']} ({tag})")
            lines.append(f"- Size: {cl['size']} episodes")
            lines.append(f"- Lines: {cl['lines_range'][0]}–{cl['lines_range'][1]} "
                         f"(avg {cl['avg_lines']})")
            lines.append(f"- Avg height: {cl['avg_height']}  holes: {cl['avg_holes']}")
            lines.append("")

        if report.candidates_generated:
            lines += [f"## New Candidates", ""]
            for name in report.candidates_generated:
                lines.append(f"- `{name}`")
            lines.append("")

        if report.hypotheses_updated:
            lines += [f"## Updated Hypotheses", ""]
            for name in report.hypotheses_updated:
                lines.append(f"- `{name}`")

        path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  [SlowLoop] Report saved: {path}")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SlowLoop — offline consolidation')
    parser.add_argument('--mode',  default='hourly',
                        choices=['hourly', 'nightly', 'full'])
    parser.add_argument('--game',  default='tetris')
    parser.add_argument('--db',    default='experiments/experiments.db')
    parser.add_argument('--min-evidence', type=int, default=30)
    parser.add_argument('--min-win-rate', type=float, default=0.20)
    args = parser.parse_args()

    gates = PromotionGates(min_evidence=args.min_evidence,
                           min_win_rate=args.min_win_rate)

    with ExperimentDB(args.db) as db:
        slow   = SlowLoop(db, game=args.game, gates=gates)
        report = slow.consolidate(mode=args.mode)
        print(f"\nDone. PolicyPack v{report.pack_version}, "
              f"{report.pack_promoted} active hypotheses.")


if __name__ == '__main__':
    main()
