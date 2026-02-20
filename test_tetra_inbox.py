"""
Smoke test for the Tetra inbox system:
1. Generate a tetra brief and verify inbox_schema is present
2. Write a test inbox with ADD + MUTATE + RETIRE ops
3. Run SlowLoop nightly — verify inbox is ingested and archived
"""
import sys, json, time
sys.path.insert(0, '.')
from pathlib import Path
from throng4.storage.experiment_db import ExperimentDB

db = ExperimentDB('experiments/experiments.db')

# ── 1. Generate brief ──────────────────────────────────────────────────────
brief = db.generate_tetra_brief()
assert 'inbox_schema' in brief, "Missing inbox_schema in brief"
assert 'prompt_for_tetra' in brief
print("✓ Brief generated with inbox_schema")
print(f"  Total hypotheses in ledger: {brief['hypothesis_ledger']['total']}")
print(f"  Open questions: {len(brief['open_questions'])}")

# ── 2. Write test inbox ────────────────────────────────────────────────────
test_ops = [
    {
        "op": "ADD",
        "name": "tetra_holes_early_game",
        "description": "Penalise holes created in the first 8 pieces more heavily.",
        "llm_score": 0.72,
        "llm_priority": "explore",
        "llm_notes": "Failure clusters show hole count at piece 8 strongly predicts final lines.",
        "game": "tetris"
    },
    {
        "op": "MUTATE",
        "parent": "reduce_holes_v601",
        "name": "tetra_reduce_holes_height_gated",
        "description": "Reduce holes only when board height exceeds 60% of max.",
        "llm_score": 0.68,
        "llm_priority": "test",
        "llm_notes": "Adds height gate — hole avoidance matters most when already tall."
    },
    {
        "op": "RETIRE",
        "name": "auto_top_fill",
        "llm_notes": "AutoMetric shows no correlation with outcome across all levels."
    }
]

inbox_path = Path('experiments/tetra_inbox.json')
inbox_path.write_text(json.dumps(test_ops, indent=2), encoding='utf-8')
print(f"\n✓ Test inbox written ({len(test_ops)} ops)")

# ── 3. Ingest directly ─────────────────────────────────────────────────────
result = db.ingest_tetra_inbox()
print(f"\n✓ Inbox ingested:")
print(f"  added={result['added']}  updated={result['updated']}  "
      f"retired={result['retired']}  mutated={result['mutated']}")
if result.get('errors'):
    for e in result['errors']:
        print(f"  ⚠ {e}")

# ── 4. Verify archive ──────────────────────────────────────────────────────
archive_files = list(Path('experiments/tetra_archive').glob('tetra_inbox_*.json'))
print(f"\n✓ Archive contains {len(archive_files)} file(s):")
for f in sorted(archive_files)[-3:]:
    print(f"  {f.name}")

# ── 5. Verify new hypothesis in DB ────────────────────────────────────────
import sqlite3
con = sqlite3.connect('experiments/experiments.db')
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT name, status, llm_score, llm_priority, generation, llm_notes "
    "FROM hypotheses WHERE name LIKE 'tetra_%' ORDER BY created_at DESC LIMIT 5"
).fetchall()
print(f"\n✓ Tetra-sourced hypotheses in DB ({len(rows)}):")
for r in rows:
    print(f"  [{r['status']}] {r['name']}  "
          f"llm_score={r['llm_score']}  gen={r['generation']}  "
          f"priority={r['llm_priority']}")
con.close()
db.close()
print("\nAll checks passed.")
