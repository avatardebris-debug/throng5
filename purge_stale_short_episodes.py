"""
purge_stale_short_episodes.py

Removes episodes with pieces_placed <= 2 that were recorded before
the game-over bug was fixed. These are physically impossible on the
current env (0/20000 reproduction rate) and poison the failure clusters
and Tetra brief with false "early-death" patterns.

Cutoff: 2026-02-19 00:00 CST (start of today).
Only episodes BEFORE this date with pieces_placed <= 2 are removed.
Episodes from today onward are kept regardless of piece count.

Usage:
    python purge_stale_short_episodes.py           # dry-run (safe)
    python purge_stale_short_episodes.py --commit  # actually delete
"""

import sqlite3
import time
import argparse

import time as _time
# Midnight at the start of today (local time) — everything before this is stale
CUTOFF_EPOCH = _time.mktime(_time.strptime('2026-02-19 00:00:00', '%Y-%m-%d %H:%M:%S'))
MAX_PIECES   = 2            # purge pieces_placed <= this value

DB_PATH = 'experiments/experiments.db'

parser = argparse.ArgumentParser()
parser.add_argument('--commit', action='store_true',
                    help='Actually delete (default is dry-run)')
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Preview what will be deleted
rows = conn.execute(
    'SELECT level, pieces_placed, COUNT(*) as n '
    'FROM episodes '
    'WHERE pieces_placed <= ? AND timestamp < ? '
    'GROUP BY level, pieces_placed '
    'ORDER BY level, pieces_placed',
    (MAX_PIECES, CUTOFF_EPOCH)
).fetchall()

total = sum(r['n'] for r in rows)
print(f'{"DRY RUN — " if not args.commit else ""}Stale episodes to purge (pieces<={MAX_PIECES}, before Feb 19):')
for r in rows:
    print(f'  L{r["level"]}  pieces={r["pieces_placed"]}  n={r["n"]}')
print(f'  TOTAL: {total} episodes')

total_before = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'\nDB before: {total_before} total episodes')

if not args.commit:
    print('\nDry run complete. Re-run with --commit to delete.')
    conn.close()
    exit(0)

# Commit the delete
conn.execute(
    'DELETE FROM episodes WHERE pieces_placed <= ? AND timestamp < ?',
    (MAX_PIECES, CUTOFF_EPOCH)
)
conn.commit()

total_after = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'DB after:  {total_after} total episodes')
print(f'Deleted:   {total_before - total_after} rows')
conn.close()
print('\nDone. Re-run generate_tetra_brief.py to refresh the brief.')
