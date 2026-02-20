"""
Generate a fresh tetra_brief.json for sending to an LLM.

Usage:
    python generate_tetra_brief.py

Then:
    1. Read experiments/TETRA_PROMPT.md for the system prompt
    2. Paste experiments/tetra_brief.json as the user message
    3. Save the LLM's JSON response to experiments/tetra_inbox.json
    4. SlowLoop will ingest it on the next nightly run
"""
import sys, json
sys.path.insert(0, '.')
from throng4.storage.experiment_db import ExperimentDB
from pathlib import Path

db = ExperimentDB('experiments/experiments.db')
brief = db.generate_tetra_brief()
db.close()

out = Path('experiments/tetra_brief.json')
out.write_text(json.dumps(brief, indent=2), encoding='utf-8')

size_kb = out.stat().st_size / 1024
hyps    = brief['hypothesis_ledger']['total']
eps     = brief['game_context']['total_episodes']
levels  = len(brief['game_context']['levels_trained'])

print(f"tetra_brief.json written ({size_kb:.1f} KB)")
print(f"  Episodes: {eps:,}   Levels: {levels}   Hypotheses: {hyps}")
print(f"  Open questions: {len(brief['open_questions'])}")
print()
print("Next steps:")
print("  1. Copy system prompt from experiments/TETRA_PROMPT.md")
print("  2. Paste experiments/tetra_brief.json as user message")
print("  3. Save LLM response to experiments/tetra_inbox.json")
print("  4. Run: python -m throng4.runners.slow_loop --mode nightly")
