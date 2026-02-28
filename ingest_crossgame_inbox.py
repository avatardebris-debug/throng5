"""
ingest_crossgame_inbox.py
==========================
Phase 3: Ingest Tetra's response to the cross-game brief.

Processes:
  - ADD / RETIRE / MUTATE ops  → route to per-game atari_hypotheses.json (existing logic)
  - PRINCIPLE ops               → write to experiments/principles.json (PrincipleStore)

Usage:
    python ingest_crossgame_inbox.py [--inbox experiments/crossgame_inbox.json]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


# Direct JSON store — avoids importing throng4 package (which conflicts with
# the running round-robin process locking .pyc files on Windows).
_PRINCIPLES_PATH = _ROOT / "experiments" / "principles.json"

def _load_principles() -> list:
    if _PRINCIPLES_PATH.exists():
        try:
            return json.loads(_PRINCIPLES_PATH.read_text(encoding="utf-8")).get("principles", [])
        except Exception:
            return []
    return []

def _save_principles(principles: list) -> None:
    _PRINCIPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PRINCIPLES_PATH.write_text(json.dumps({"principles": principles}, indent=2), encoding="utf-8")

def _ingest_principle(op: dict) -> bool:
    principles = _load_principles()
    existing = {p["id"]: i for i, p in enumerate(principles)}
    entry = {
        "id":          op["id"],
        "text":        op["text"],
        "source":      "tetra",
        "env_class":   op.get("env_class", {}),
        "params":      op.get("params", {}),
        "evidence":    op.get("evidence", []),
        "confidence":  float(op.get("confidence", 0.5)),
        "created_at":  time.time(),
        "updated_at":  time.time(),
    }
    if entry["id"] in existing:
        principles[existing[entry["id"]]] = entry
    else:
        principles.append(entry)
    _save_principles(principles)
    return True

_HYPS_PATH   = _ROOT / "experiments" / "atari_hypotheses.json"
_DEFAULT_INBOX = _ROOT / "experiments" / "crossgame_inbox.json"


def _load_hyps() -> dict:
    if _HYPS_PATH.exists():
        try:
            return json.loads(_HYPS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active": [], "retired": [], "candidates": []}


def _save_hyps(data: dict) -> None:
    _HYPS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ingest_game_op(op: dict, hyps: dict) -> bool:
    """Route ADD/RETIRE/MUTATE to the hypothesis ledger (same schema as atari_inbox)."""
    op_type = op.get("op", "").upper()
    now     = time.time()

    if op_type == "ADD":
        entry = {k: v for k, v in op.items() if k != "op"}
        entry.setdefault("status",       "active")
        entry.setdefault("evidence_count", 0)
        entry.setdefault("source",       "crossgame_inbox")
        entry["created_at"] = now
        entry["updated_at"] = now
        hyps.setdefault("active", []).append(entry)
        print(f"  [ADD]    {entry.get('name','?')}  game={entry.get('game','?')}")
        return True

    if op_type == "RETIRE":
        name = op.get("name", "")
        for section in ("active", "candidates"):
            for i, h in enumerate(hyps.get(section, [])):
                if h.get("name") == name:
                    h["status"]     = "retired"
                    h["llm_notes"]  = op.get("llm_notes", "")
                    h["updated_at"] = now
                    hyps.setdefault("retired", []).append(hyps[section].pop(i))
                    print(f"  [RETIRE] {name}")
                    return True

    if op_type == "MUTATE":
        parent_name = op.get("parent", "")
        entry = {k: v for k, v in op.items() if k not in ("op", "parent")}
        entry["source"]        = "crossgame_mutation"
        entry["parent_name"]   = parent_name
        entry["created_at"]    = now
        entry["updated_at"]    = now
        entry["evidence_count"] = 0
        hyps.setdefault("active", []).append(entry)
        print(f"  [MUTATE] {entry.get('name','?')} (from {parent_name})")
        return True

    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inbox", default=str(_DEFAULT_INBOX))
    args = p.parse_args()

    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(f"[ingest] Inbox not found: {inbox_path}")
        print("  Paste Tetra's JSON array into that file, then re-run.")
        return

    try:
        ops = json.loads(inbox_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ingest] JSON parse error: {e}")
        return

    if not isinstance(ops, list):
        print("[ingest] Expected a JSON array of ops")
        return

    hyps = _load_hyps()
    n_game     = 0
    n_principle = 0
    n_failed    = 0

    for op in ops:
        op_type = str(op.get("op","")).upper()
        if op_type == "PRINCIPLE":
            ok = _ingest_principle(op)
            if ok:
                n_principle += 1
                print(f"  [PRINCIPLE] {op.get('id','?')}  env={op.get('env_class',{})}  "
                      f"conf={op.get('confidence',0):.2f}")
            else:
                n_failed += 1
        elif op_type in ("ADD", "RETIRE", "MUTATE"):
            ok = _ingest_game_op(op, hyps)
            if ok:
                n_game += 1
            else:
                n_failed += 1
                print(f"  [SKIP] unknown op or target: {op}")
        else:
            print(f"  [SKIP] unrecognised op type: {op_type}")
            n_failed += 1

    if n_game > 0:
        _save_hyps(hyps)

    print(f"\n[ingest] Done: {n_game} game ops  {n_principle} principles  "
          f"{n_failed} failed/skipped")
    if n_principle > 0:
        all_p = _load_principles()
        print(f"[ingest] principles.json now has {len(all_p)} entries total")


if __name__ == "__main__":
    main()
