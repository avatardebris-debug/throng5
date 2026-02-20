"""
patch_tetra_enactions.py — One-shot: add enaction schemas to Tetra's first run hypotheses.

Tetra generated l7_opening_hole_cap_12, l7_bumpiness_guardrail, and
tetra_survive_tall_boards_hole_cap_opening_gate without enaction metadata
(it didn't know the schema yet). This patches their metadata so the
EnactionEngine can act on them at the next SlowLoop promotion.

Run once:
    python patch_tetra_enactions.py
"""

import json
import sqlite3
import time

DB_PATH = 'experiments/experiments.db'

PATCHES = [
    {
        'name': 'l7_opening_hole_cap_12',
        'game': 'tetris',
        'enaction': {
            'type':       'piece_phase',
            'range':      [0, 12],
            'target':     'holes',
            'multiplier': 2.0,
        },
        'note': 'Opening phase: double the hole penalty for pieces 0-12',
    },
    {
        'name': 'l7_bumpiness_guardrail',
        'game': 'tetris',
        'enaction': {
            'type':       'reward_weight',
            'target':     'bumpiness',
            'multiplier': 1.8,
        },
        'note': 'Always-on: 1.8x bumpiness penalty to keep surface smooth',
    },
    {
        'name': 'tetra_survive_tall_boards_hole_cap_opening_gate',
        'game': 'tetris',
        'enaction': {
            'type':       'piece_phase',
            'range':      [0, 12],
            'target':     'holes',
            'multiplier': 2.5,
        },
        'note': 'Stricter variant: 2.5x hole penalty in opening phase',
    },
]

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()
now = time.time()

for patch in PATCHES:
    row = c.execute(
        'SELECT id, metadata FROM hypotheses WHERE name=? AND game=?',
        (patch['name'], patch['game'])
    ).fetchone()

    if not row:
        print(f"  ✗ Not found: {patch['name']} — skipping")
        continue

    # Merge enaction into existing metadata
    existing_meta = {}
    if row['metadata']:
        try:
            existing_meta = json.loads(row['metadata'])
        except Exception:
            pass

    existing_meta['enaction']        = patch['enaction']
    existing_meta['enaction_note']   = patch['note']
    existing_meta['enaction_patched'] = True

    c.execute(
        'UPDATE hypotheses SET metadata=?, updated_at=? WHERE id=?',
        (json.dumps(existing_meta), now, row['id'])
    )
    print(f"  ✓ Patched {patch['name']}: {patch['enaction']['type']} "
          f"target={patch['enaction'].get('target')} "
          f"mult={patch['enaction'].get('multiplier')}")

conn.commit()
conn.close()
print("\nDone. Run SlowLoop to promote and write active_enactions.json.")
