"""
validate_blind_hypotheses.py — One-command gate for blind hypothesis JSON.

Pipeline:
  1. Load hypotheses JSON from Tetra
  2. Run 4-layer validation:
       a. Required keys   (incl. generality, trigger)
       b. Enum validity   (generality, direction)
       c. Confidence range [0, 1]
       d. Abstract-vocab lint (reject game-specific terms)
  3. Print detailed report
  4. If clean: ingest into RuleLibrary + snapshot to SQLite restart archive
  5. Exit 0 on success, 1 on validation failure

Usage:
    # Validate + ingest (typical workflow):
    python validate_blind_hypotheses.py ~/.openclaw/workspace/memory/hypotheses_XYZ.json tetris

    # Validate only, no ingest:
    python validate_blind_hypotheses.py hypotheses.json tetris --dry-run

    # Show full report even if valid:
    python validate_blind_hypotheses.py hypotheses.json tetris --verbose
"""

import json
import sys
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from throng4.config import (
    REQUIRED_HYPOTHESIS_KEYS,
    VALID_GENERALITY_VALUES,
    VALID_DIRECTION_VALUES,
    BLIND_GAME_STRINGS,
)


# ---------------------------------------------------------------------------
# Validation layers
# ---------------------------------------------------------------------------

ValidationResult = Tuple[bool, List[str]]   # (passed, error_messages)


def validate_required_keys(h: dict, idx: int) -> ValidationResult:
    """Layer 1: All required keys must be present."""
    missing = REQUIRED_HYPOTHESIS_KEYS - set(h.keys())
    if missing:
        return False, [f"[{idx}] missing keys: {sorted(missing)}"]
    return True, []


def validate_enums(h: dict, idx: int) -> ValidationResult:
    """Layer 2: Enum fields must have allowed values."""
    errors = []
    gen = h.get("generality", "")
    if gen not in VALID_GENERALITY_VALUES:
        errors.append(
            f"[{idx}] 'generality' = {gen!r} — "
            f"must be one of {sorted(VALID_GENERALITY_VALUES)}"
        )
    direction = h.get("direction", "")
    if direction not in VALID_DIRECTION_VALUES:
        errors.append(
            f"[{idx}] 'direction' = {direction!r} — "
            f"must be one of {sorted(VALID_DIRECTION_VALUES)}"
        )
    return len(errors) == 0, errors


def validate_confidence(h: dict, idx: int) -> ValidationResult:
    """Layer 3: Confidence must be a float in [0.0, 1.0]."""
    try:
        conf = float(h.get("confidence", -1))
    except (TypeError, ValueError):
        return False, [f"[{idx}] 'confidence' is not a number: {h.get('confidence')!r}"]
    if not (0.0 <= conf <= 1.0):
        return False, [f"[{idx}] 'confidence' = {conf} out of range [0, 1]"]
    return True, []


def validate_abstract_vocab(h: dict, idx: int) -> ValidationResult:
    """
    Layer 4: Required fields must not contain game-specific terms.
    Checks: description, object, feature, trigger.
    """
    errors = []
    for field in ("description", "object", "feature", "trigger"):
        val = str(h.get(field, ""))
        leaks = [s for s in BLIND_GAME_STRINGS if s in val]
        if leaks:
            errors.append(
                f"[{idx}] '{field}' contains game-specific terms: {leaks}  "
                f"(value: {val[:80]!r})"
            )
    return len(errors) == 0, errors


def run_all_layers(hypotheses: List[Dict[str, Any]]) -> Tuple[bool, List[str], Dict]:
    """
    Run all 4 validation layers on all hypotheses.

    Returns:
        (all_passed, all_errors, stats_dict)
    """
    all_errors: List[str] = []
    stats = {
        "total":            len(hypotheses),
        "key_errors":       0,
        "enum_errors":      0,
        "confidence_errors": 0,
        "vocab_errors":     0,
        "generality_dist":  {"universal": 0, "class": 0, "instance": 0, "invalid": 0},
        "direction_dist":   {},
        "valid_count":      0,
    }

    for i, h in enumerate(hypotheses):
        h_ok = True

        ok, errs = validate_required_keys(h, i)
        if not ok:
            all_errors.extend(errs)
            stats["key_errors"] += 1
            h_ok = False
            continue  # skip remaining checks if keys missing

        ok, errs = validate_enums(h, i)
        if not ok:
            all_errors.extend(errs)
            stats["enum_errors"] += 1
            h_ok = False

        ok, errs = validate_confidence(h, i)
        if not ok:
            all_errors.extend(errs)
            stats["confidence_errors"] += 1
            h_ok = False

        ok, errs = validate_abstract_vocab(h, i)
        if not ok:
            all_errors.extend(errs)
            stats["vocab_errors"] += 1
            h_ok = False

        # Tallies (even for invalid hypotheses where key check passed)
        gen = h.get("generality", "invalid")
        stats["generality_dist"][gen if gen in VALID_GENERALITY_VALUES else "invalid"] += 1

        direction = h.get("direction", "")
        stats["direction_dist"][direction] = stats["direction_dist"].get(direction, 0) + 1

        if h_ok:
            stats["valid_count"] += 1

    return len(all_errors) == 0, all_errors, stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(hypotheses: List[dict], errors: List[str],
                 stats: Dict, passed: bool) -> None:
    w = 58
    print(f"\n{'='*w}")
    print(f"  Blind Hypothesis Validation Report")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*w}")
    print(f"  Total hypotheses : {stats['total']}")
    print(f"  Valid            : {stats['valid_count']}")
    print(f"  Key errors       : {stats['key_errors']}")
    print(f"  Enum errors      : {stats['enum_errors']}")
    print(f"  Confidence errors: {stats['confidence_errors']}")
    print(f"  Vocab errors     : {stats['vocab_errors']}")
    print()
    print("  Generality distribution:")
    for k, v in sorted(stats["generality_dist"].items()):
        if v > 0:
            pct = v / stats["total"] * 100
            bar = "█" * v
            print(f"    {k:<12} {v:2d} ({pct:4.0f}%)  {bar}")
    print()
    print("  Direction distribution:")
    for k, v in sorted(stats["direction_dist"].items()):
        print(f"    {k:<12} {v:2d}")

    if errors:
        print(f"\n  ── Errors ({len(errors)}) ──")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("\n  ── All checks passed ──")

    print()
    for i, h in enumerate(hypotheses):
        valid_marker = "✓" if (
            validate_required_keys(h, i)[0] and
            validate_enums(h, i)[0] and
            validate_confidence(h, i)[0] and
            validate_abstract_vocab(h, i)[0]
        ) else "✗"
        conf = h.get("confidence", "?")
        gen  = h.get("generality", "?")
        feat = h.get("feature", "?")[:20]
        desc = h.get("description", "?")[:55]
        print(f"  [{valid_marker}] {i+1}. [{gen:<9}] {feat:<22} {desc}")

    result_str = "✅ PASSED — ready to ingest" if passed else "❌ FAILED — not ingested"
    print(f"\n  {result_str}")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# SQLite snapshot
# ---------------------------------------------------------------------------

def snapshot_to_db(db_path: str, game_id: str, blind_label: str,
                   source_file: str, stats: Dict,
                   hypotheses: List[dict]) -> None:
    """
    Write a short validation summary to the restart archive in SQLite.
    Table: blind_hypothesis_log (created on first use, append-only).
    """
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS blind_hypothesis_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT NOT NULL,
            game_id      TEXT NOT NULL,
            blind_label  TEXT NOT NULL,
            source_file  TEXT NOT NULL,
            total        INTEGER,
            valid_count  INTEGER,
            key_errors   INTEGER,
            enum_errors  INTEGER,
            conf_errors  INTEGER,
            vocab_errors INTEGER,
            gen_universal INTEGER,
            gen_class     INTEGER,
            gen_instance  INTEGER,
            hypotheses_json TEXT
        )
    """)
    ts = datetime.now(timezone.utc).isoformat()
    db.execute("""
        INSERT INTO blind_hypothesis_log
          (ts, game_id, blind_label, source_file,
           total, valid_count, key_errors, enum_errors,
           conf_errors, vocab_errors,
           gen_universal, gen_class, gen_instance, hypotheses_json)
        VALUES (?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?)
    """, (
        ts, game_id, blind_label, source_file,
        stats["total"], stats["valid_count"],
        stats["key_errors"], stats["enum_errors"],
        stats["confidence_errors"], stats["vocab_errors"],
        stats["generality_dist"].get("universal", 0),
        stats["generality_dist"].get("class", 0),
        stats["generality_dist"].get("instance", 0),
        json.dumps(hypotheses),
    ))
    db.commit()
    db.close()
    print(f"  📦 Snapshot written → {db_path}  (table: blind_hypothesis_log)")


# ---------------------------------------------------------------------------
# Ingest into RuleLibrary
# ---------------------------------------------------------------------------

def ingest(hypotheses: List[dict], game_id: str) -> int:
    """Ingest valid hypotheses into the RuleLibrary and persist."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    gen = OfflineGenerator(game_id=game_id)
    before = len(gen.library.rules)
    gen._add_hypotheses_to_library(hypotheses)
    after = len(gen.library.rules)
    gen.save_library()
    added = after - before
    from throng4.config import rules_path
    print(f"  📚 Ingested {added} new rule(s) → {rules_path(game_id)}")
    return added


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate and ingest blind hypotheses from Tetra"
    )
    parser.add_argument("hyp_file",  help="Path to hypotheses JSON from Tetra")
    parser.add_argument("game_id",   help="Game identifier (e.g. tetris, ALE/Pong-v5)")
    parser.add_argument("--db",      default="experiments/experiments.db",
                        help="SQLite DB path for snapshot (default: experiments/experiments.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only; do not ingest or snapshot")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full report even when all checks pass")
    args = parser.parse_args()

    # -- Load
    hyp_path = Path(args.hyp_file)
    if not hyp_path.exists():
        print(f"❌ File not found: {hyp_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(hyp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}", file=sys.stderr)
        sys.exit(1)

    hypotheses = data.get("hypotheses", [])
    if not hypotheses:
        print("❌ No 'hypotheses' key or empty list in response.", file=sys.stderr)
        sys.exit(1)

    # -- Validate
    passed, errors, stats = run_all_layers(hypotheses)

    if not passed or args.verbose:
        print_report(hypotheses, errors, stats, passed)
    else:
        # Compact summary when all clean
        gen_dist = stats["generality_dist"]
        print(
            f"\n✅  {stats['valid_count']}/{stats['total']} hypotheses valid  "
            f"[U={gen_dist['universal']} C={gen_dist['class']} I={gen_dist['instance']}]  "
            f"— ingesting…"
        )

    if not passed:
        print(f"  Found {len(errors)} error(s). Fix before ingesting.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("  [dry-run] Skipping ingest + snapshot.")
        sys.exit(0)

    # -- Blind label for snapshot
    from throng4.llm_policy.offline_generator import _get_blind_label
    blind_label = _get_blind_label(args.game_id)

    # -- Ingest
    added = ingest(hypotheses, args.game_id)

    # -- Snapshot
    snapshot_to_db(
        db_path=args.db,
        game_id=args.game_id,
        blind_label=blind_label,
        source_file=str(hyp_path.resolve()),
        stats=stats,
        hypotheses=hypotheses,
    )

    print(f"\n  Done. {added} rule(s) added for {args.game_id} ({blind_label})")


if __name__ == "__main__":
    main()
