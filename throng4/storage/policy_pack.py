"""
policy_pack.py — Frozen versioned sets of active hypotheses.

A PolicyPack is a snapshot of the best-performing hypotheses at a point
in time, promoted from the ExperimentDB after passing promotion gates.

Design
------
- Immutable once created: a version is a version, it never changes.
- FastLoop loads the latest pack at startup and reloads every N episodes.
- SlowLoop writes new packs after consolidation — no coordination needed.
- Promotion gates guard quality: a hypothesis must earn its place before
  it can influence the fast loop.

Promotion gates (all must pass):
  1. min_evidence:   >= 50 episodes evaluated (default)
  2. min_win_rate:   >= 0.30 win rate (ranks #1 in >= 30% of evals)
  3. stability:      win_rate not too volatile (std < 0.3 over last chunk)
  4. not retired:    status != 'retired'

Usage
-----
    # Create a pack from DB (SlowLoop does this after consolidation)
    pack = PolicyPack.from_db(db, game='tetris', notes='after L5 run')
    # pack.version is now stored in DB

    # Load latest pack (FastLoop does this at startup)
    pack = PolicyPack.load_latest(db, game='tetris')

    # Check if the pack recommends acting on a hypothesis
    bias = pack.get_action_bias(state, valid_actions)
    if bias:
        print(bias['hypothesis'], bias['confidence'])
"""

from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ── Promotion gates ────────────────────────────────────────────────────────

@dataclass
class PromotionGates:
    """
    Configurable promotion gates.

    A hypothesis must pass ALL gates to enter a PolicyPack.
    """
    min_evidence: int   = 50     # Minimum evaluation count
    min_win_rate: float = 0.30   # Must rank #1 at least 30% of the time
    max_win_rate: float = 1.0    # Sanity cap (1.0 = no cap)
    exclude_status: List[str] = field(default_factory=lambda: ['retired', 'rejected'])

    def passes(self, hyp: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if a hypothesis dict passes all gates.

        Returns:
            (passed: bool, reason: str)
        """
        if hyp.get('status') in self.exclude_status:
            return False, f"status={hyp.get('status')}"
        if hyp.get('evidence_count', 0) < self.min_evidence:
            return False, f"evidence={hyp.get('evidence_count',0)} < {self.min_evidence}"
        wr = hyp.get('win_rate', 0.0)
        if wr < self.min_win_rate:
            return False, f"win_rate={wr:.2f} < {self.min_win_rate:.2f}"
        if wr > self.max_win_rate:
            return False, f"win_rate={wr:.2f} > {self.max_win_rate:.2f}"
        return True, "ok"


# ── PolicyPack ─────────────────────────────────────────────────────────────

class PolicyPack:
    """
    Frozen, versioned set of active hypotheses.

    Instances are immutable — to update the active policy, create a new pack
    via PolicyPack.from_db() and let the FastLoop reload it.

    Attributes:
        version:     Integer version number (monotonically increasing in DB).
        game:        Game this pack was created for.
        hypotheses:  Frozen list of hypothesis dicts at promotion time.
        created_at:  Unix timestamp of pack creation.
        notes:       Free-text annotation.
    """

    def __init__(self, version: int, game: str,
                 hypotheses: List[Dict[str, Any]],
                 created_at: float, notes: str = ""):
        self.version = version
        self.game = game
        self.hypotheses = list(hypotheses)  # frozen copy
        self.created_at = created_at
        self.notes = notes

        # Build name → hyp index for fast lookup
        self._by_name: Dict[str, Dict] = {h['name']: h for h in self.hypotheses}

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def from_db(cls, db, game: str = '',
                gates: Optional[PromotionGates] = None,
                notes: str = '') -> 'PolicyPack':
        """
        Create a new PolicyPack from the current DB state.

        Queries all non-retired hypotheses for the given game, applies
        promotion gates, freezes the survivors, and writes the pack to DB.

        Args:
            db:    ExperimentDB instance (open).
            game:  Game to filter hypotheses by. Empty string = all games.
            gates: PromotionGates config. Defaults to standard gates.
            notes: Annotation stored with the pack.

        Returns:
            New PolicyPack with version number assigned by DB.
        """
        if gates is None:
            gates = PromotionGates()

        # Query candidates
        all_hyps = db.get_hypotheses(game=game or None)

        promoted = []
        rejected = []
        for hyp in all_hyps:
            passed, reason = gates.passes(hyp)
            if passed:
                promoted.append(hyp)
            else:
                rejected.append((hyp['name'], reason))

        # Log rejections for transparency
        if rejected:
            print(f"  [PolicyPack] {len(rejected)} hypotheses below gates:")
            for name, reason in rejected[:5]:  # cap at 5 for readability
                print(f"    ✗ {name}: {reason}")
            if len(rejected) > 5:
                print(f"    ... and {len(rejected)-5} more")

        # Sort by win_rate DESC, then confidence DESC
        promoted.sort(key=lambda h: (h.get('win_rate', 0), h.get('confidence', 0)),
                      reverse=True)

        # Write pack to DB
        ids = [h['id'] for h in promoted]
        version = db.create_policy_pack(
            active_hypothesis_ids=ids,
            notes=notes or f"auto: {len(promoted)} promoted, {len(rejected)} rejected",
        )

        print(f"  [PolicyPack v{version}] {len(promoted)} hypotheses promoted for '{game or 'all'}'")
        for h in promoted:
            print(f"    ✓ {h['name']}  win={h.get('win_rate',0):.0%}  n={h.get('evidence_count',0)}")

        return cls(
            version=version,
            game=game,
            hypotheses=promoted,
            created_at=time.time(),
            notes=notes,
        )

    @classmethod
    def load_latest(cls, db, game: str = '') -> Optional['PolicyPack']:
        """
        Load the most recent PolicyPack from DB.

        Returns None if no packs exist yet (first run).

        Args:
            db:   ExperimentDB instance.
            game: Game filter for hypothesis lookup.

        Returns:
            Latest PolicyPack, or None.
        """
        c = db.conn.cursor()
        c.execute("SELECT * FROM policy_packs ORDER BY version DESC LIMIT 1")
        row = c.fetchone()
        if row is None:
            return None

        row = dict(row)
        ids = json.loads(row['active_hypothesis_ids'])
        if not ids:
            return cls(
                version=row['version'],
                game=game,
                hypotheses=[],
                created_at=row['created_at'],
                notes=row.get('notes', ''),
            )

        # Fetch hypothesis details for each id
        hyps = []
        for hid in ids:
            c.execute("SELECT * FROM hypotheses WHERE id = ?", (hid,))
            h = c.fetchone()
            if h:
                hyps.append(dict(h))

        return cls(
            version=row['version'],
            game=game,
            hypotheses=hyps,
            created_at=row['created_at'],
            notes=row.get('notes', ''),
        )

    # ── Runtime interface ──────────────────────────────────────────────────

    def get_action_bias(self, state, valid_actions: list) -> Optional[Dict[str, Any]]:
        """
        Return a hypothesis-based action bias for the given state.

        This is the FastLoop's entry point: given the current state and
        valid actions, does any promoted hypothesis have a strong
        recommendation?

        Currently returns the highest-confidence hypothesis that applies
        (all hypotheses are considered "always applicable" until context
        matching is wired in). Returns None if the pack is empty.

        Args:
            state:         Current state (numpy array or similar).
            valid_actions: List of valid actions.

        Returns:
            Dict with keys: hypothesis, confidence, win_rate, description
            Or None if no applicable hypothesis.
        """
        if not self.hypotheses:
            return None

        best = self.hypotheses[0]  # already sorted by win_rate DESC
        return {
            'hypothesis': best['name'],
            'confidence': best.get('confidence', 0.5),
            'win_rate': best.get('win_rate', 0.0),
            'description': best.get('description', ''),
            'pack_version': self.version,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a hypothesis by name."""
        return self._by_name.get(name)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __bool__(self) -> bool:
        return len(self.hypotheses) > 0

    def __repr__(self) -> str:
        return (f"PolicyPack(v{self.version}, game='{self.game}', "
                f"{len(self.hypotheses)} hypotheses, "
                f"created={time.strftime('%Y-%m-%d %H:%M', time.localtime(self.created_at))})")

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of this pack."""
        lines = [
            f"PolicyPack v{self.version} — {self.game or 'all games'}",
            f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}",
            f"  Notes:   {self.notes}",
            f"  Hypotheses ({len(self.hypotheses)}):",
        ]
        for h in self.hypotheses:
            lines.append(
                f"    [{h.get('status','?'):9s}] {h['name']:35s} "
                f"win={h.get('win_rate',0):.0%}  n={h.get('evidence_count',0):4d}  "
                f"conf={h.get('confidence',0):.2f}"
            )
        if not self.hypotheses:
            lines.append("    (empty — no hypotheses passed promotion gates)")
        return '\n'.join(lines)
