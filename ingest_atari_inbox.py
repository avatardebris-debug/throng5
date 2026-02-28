"""
ingest_atari_inbox.py
=====================
Cron step 6 — ingest the LLM's Atari hypothesis ops from atari_inbox.json.

Reads:   experiments/atari_inbox.json   (JSON array of ADD/RETIRE/MUTATE ops)
Updates: experiments/atari_hypotheses.json   (persistent ledger)
Exports: experiments/atari_active_ops.json   (training scripts read this)
Archives: experiments/tetra_archive/atari_inbox_<timestamp>.json

Usage:
    python ingest_atari_inbox.py
    python ingest_atari_inbox.py --inbox experiments/atari_inbox.json
    python ingest_atari_inbox.py --hyps experiments/atari_hypotheses.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT         = Path(__file__).resolve().parent
_EXP          = _ROOT / "experiments"
_INBOX_PATH   = _EXP / "atari_inbox.json"
_HYPS_PATH    = _EXP / "atari_hypotheses.json"
_ACTIVE_PATH  = _EXP / "atari_active_ops.json"
_ARCHIVE_DIR  = _EXP / "tetra_archive"

_REQUIRED_ADD    = {"op", "name", "description", "game", "llm_score",
                    "llm_priority", "llm_notes", "enaction"}
_REQUIRED_RETIRE = {"op", "name", "llm_notes"}
_REQUIRED_MUTATE = {"op", "parent", "name", "description", "game",
                    "llm_score", "llm_priority", "llm_notes", "enaction"}


# ──────────────────────────────────────────────────────────────────────
# Load / save helpers
# ──────────────────────────────────────────────────────────────────────

def _load_hyps(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[atari] WARN: could not parse {path}: {exc}. Starting fresh.")
    return {"active": [], "retired": [], "candidates": []}


def _save_hyps(hyps: dict, path: Path) -> None:
    path.write_text(json.dumps(hyps, indent=2), encoding="utf-8")


def _export_active_ops(hyps: dict, path: Path) -> None:
    """
    Write atari_active_ops.json — the file training scripts read.
    Contains only active hypotheses with their enaction fields.
    """
    active_ops = []
    for h in hyps.get("active", []):
        if h.get("enaction"):
            active_ops.append({
                "name":       h["name"],
                "game":       h.get("game"),
                "enaction":   h["enaction"],
                "llm_score":  h.get("llm_score", 0.5),
                "llm_priority": h.get("llm_priority", "explore"),
            })
    path.write_text(json.dumps(active_ops, indent=2), encoding="utf-8")


def _archive(inbox_path: Path) -> None:
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts   = int(time.time())
    dest = _ARCHIVE_DIR / f"atari_inbox_{ts}.json"
    dest.write_bytes(inbox_path.read_bytes())


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────

def _validate_op(op: dict) -> str | None:
    """Return an error string or None if valid."""
    kind = op.get("op", "").upper()
    if kind == "ADD":
        missing = _REQUIRED_ADD - set(op.keys())
        if missing:
            return f"ADD op '{op.get('name')}' missing fields: {missing}"
    elif kind == "RETIRE":
        missing = _REQUIRED_RETIRE - set(op.keys())
        if missing:
            return f"RETIRE op missing fields: {missing}"
    elif kind == "MUTATE":
        missing = _REQUIRED_MUTATE - set(op.keys())
        if missing:
            return f"MUTATE op '{op.get('name')}' missing fields: {missing}"
    else:
        return f"Unknown op type: '{op.get('op')}'"
    return None


# ──────────────────────────────────────────────────────────────────────
# Op processors
# ──────────────────────────────────────────────────────────────────────

def _apply_add(op: dict, hyps: dict) -> str:
    now = time.time()
    new_hyp = {
        "name":          op["name"],
        "game":          op.get("game"),
        "status":        "active",
        "description":   op["description"],
        "enaction":      op.get("enaction"),
        "llm_score":     op.get("llm_score", 0.5),
        "llm_priority":  op.get("llm_priority", "explore"),
        "llm_notes":     op.get("llm_notes", ""),
        "evidence_count": 0,
        "source":        "atari_inbox",
        "created_at":    now,
        "updated_at":    now,
    }
    hyps["active"].append(new_hyp)
    return f"  ✓ ADD  '{op['name']}' → active"


def _apply_retire(op: dict, hyps: dict) -> str:
    name = op["name"]
    moved = 0
    for section in ("active", "candidates"):
        remaining = []
        for h in hyps.get(section, []):
            if h["name"] == name:
                h["status"]     = "retired"
                h["retired_at"] = time.time()
                h["retire_notes"] = op.get("llm_notes", "")
                hyps.setdefault("retired", []).append(h)
                moved += 1
            else:
                remaining.append(h)
        hyps[section] = remaining
    if moved:
        return f"  ✓ RETIRE '{name}' ({moved} entry/entries moved)"
    return f"  ⚠ RETIRE '{name}' — not found in active/candidates (skipped)"


def _apply_mutate(op: dict, hyps: dict) -> str:
    now    = time.time()
    parent = op["parent"]
    # Find parent
    parent_hyp = next(
        (h for section in ("active", "candidates")
         for h in hyps.get(section, [])
         if h["name"] == parent),
        None
    )
    new_hyp = {
        "name":          op["name"],
        "game":          op.get("game"),
        "status":        "active",
        "description":   op["description"],
        "enaction":      op.get("enaction"),
        "llm_score":     op.get("llm_score", 0.5),
        "llm_priority":  op.get("llm_priority", "test"),
        "llm_notes":     op.get("llm_notes", ""),
        "evidence_count": 0,
        "source":        "atari_mutation",
        "parent_name":   parent,
        "created_at":    now,
        "updated_at":    now,
    }
    if parent_hyp:
        new_hyp["parent_enaction"] = parent_hyp.get("enaction")
    hyps["active"].append(new_hyp)
    note = "" if parent_hyp else " (parent not found — created anyway)"
    return f"  ✓ MUTATE '{parent}' → '{op['name']}'{note}"


# ──────────────────────────────────────────────────────────────────────
# Main ingest
# ──────────────────────────────────────────────────────────────────────

def ingest(inbox_path: Path, hyps_path: Path) -> dict:
    if not inbox_path.exists():
        return {"skipped": True, "reason": f"No inbox file at {inbox_path}"}

    raw = inbox_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"skipped": True, "reason": "Inbox file is empty"}

    try:
        ops = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"atari_inbox.json is not valid JSON: {exc}"}

    if not isinstance(ops, list):
        return {"error": "atari_inbox.json must be a JSON array at top level"}

    hyps   = _load_hyps(hyps_path)
    added  = retired = mutated = 0
    errors = []
    log    = []

    for op in ops:
        err = _validate_op(op)
        if err:
            errors.append(err)
            log.append(f"  ✗ {err}")
            continue

        kind = op["op"].upper()
        if kind == "ADD":
            log.append(_apply_add(op, hyps))
            added += 1
        elif kind == "RETIRE":
            msg = _apply_retire(op, hyps)
            log.append(msg)
            if "✓" in msg:
                retired += 1
        elif kind == "MUTATE":
            log.append(_apply_mutate(op, hyps))
            mutated += 1

    _save_hyps(hyps, hyps_path)
    _export_active_ops(hyps, _ACTIVE_PATH)
    _archive(inbox_path)
    inbox_path.unlink()   # consume inbox (same as Tetris ingestor)

    return {
        "added":   added,
        "retired": retired,
        "mutated": mutated,
        "errors":  errors,
        "log":     log,
        "n_active": len(hyps.get("active", [])),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Atari hypothesis inbox")
    parser.add_argument("--inbox", default=str(_INBOX_PATH))
    parser.add_argument("--hyps",  default=str(_HYPS_PATH))
    args = parser.parse_args()

    result = ingest(Path(args.inbox), Path(args.hyps))

    if result.get("skipped"):
        print(f"[atari] No inbox to ingest — {result.get('reason', '')}")
        sys.exit(0)

    if "error" in result:
        print(f"[atari] ✗ Ingest failed: {result['error']}")
        sys.exit(1)

    for line in result.get("log", []):
        print(line)

    added   = result.get("added",   0)
    retired = result.get("retired", 0)
    mutated = result.get("mutated", 0)
    errors  = result.get("errors",  [])
    print(
        f"[atari] ✓ Inbox ingested: "
        f"+{added} added  -{retired} retired  ↺{mutated} mutated  "
        f"→ {result['n_active']} active total"
    )
    print(f"[atari]   active ops exported → {_ACTIVE_PATH.name}")

    if errors:
        print(f"[atari] ⚠ {len(errors)} op error(s):")
        for e in errors:
            print(f"  ↳ {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
