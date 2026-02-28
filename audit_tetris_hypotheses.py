"""
audit_tetris_hypotheses.py
===========================
Phase 0: Tag Tetris hypotheses created before the board-corruption fix (2026-02-20)
as contaminated. These were trained on corrupted board state and may encode
false correlations.

Usage:
    python audit_tetris_hypotheses.py [--dry-run]
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path

_ROOT       = Path(__file__).resolve().parent
_HYPS_PATH  = _ROOT / "experiments" / "atari_hypotheses.json"
_CUTOFF     = "2026-02-20"   # board-corruption fix merged

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print only, don't write")
    args = p.parse_args()

    if not _HYPS_PATH.exists():
        print(f"[audit] No hypothesis file found at {_HYPS_PATH} — nothing to audit.")
        return

    data = json.loads(_HYPS_PATH.read_text(encoding="utf-8"))

    total = 0
    flagged = 0

    for section in ("active", "candidates", "retired"):
        for hyp in data.get(section, []):
            game = hyp.get("game", "")
            if "Tetris" not in game and "tetris" not in game.lower():
                continue
            total += 1
            created = hyp.get("created_at", "9999")
            if created < _CUTOFF:
                if not hyp.get("contaminated"):
                    hyp["contaminated"] = True
                    hyp["contamination_reason"] = (
                        f"Created {created} — before board-corruption fix "
                        f"({_CUTOFF}). Board state was corrupt; false correlations likely."
                    )
                    flagged += 1
                    print(f"  [FLAG] [{section}] {hyp.get('name','?')}  "
                          f"created={created}  game={game}")
            else:
                print(f"  [OK ] [{section}] {hyp.get('name','?')}  "
                      f"created={created}  game={game}")

    # Also audit tetra_brief.json if present
    tetra_path = _ROOT / "experiments" / "tetra_brief.json"
    if tetra_path.exists():
        try:
            tb = json.loads(tetra_path.read_text(encoding="utf-8"))
            tb_tetris = [h for h in tb.get("hypotheses", [])
                         if "tetris" in h.get("game","").lower()
                         and h.get("created_at","9999") < _CUTOFF]
            if tb_tetris:
                print(f"\n  [INFO] {len(tb_tetris)} pre-cutoff Tetris hypotheses "
                      f"in tetra_brief.json (informational, not modified)")
        except Exception:
            pass

    print(f"\n[audit] Tetris hypotheses total={total}  flagged={flagged}")
    if flagged == 0:
        print("[audit] Nothing to flag (either all post-cutoff or no Tetris hyps present)")

    if args.dry_run:
        print("[audit] --dry-run: no changes written")
        return

    _HYPS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[audit] Written -> {_HYPS_PATH}")

if __name__ == "__main__":
    main()
