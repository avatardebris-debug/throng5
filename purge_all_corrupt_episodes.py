"""
purge_all_corrupt_episodes.py

The make_features() bug (numpy view instead of copy) caused board corruption
throughout ALL training sessions. This script removes ALL short episodes
(pieces_placed <= 2) regardless of date, and also removes suspicious episodes
with max_height=20 AND holes > 80 (symptom of pre-filled board).

After running this, the DB will only contain valid episodes.

Run: python purge_all_corrupt_episodes.py           # dry-run
     python purge_all_corrupt_episodes.py --commit  # delete
"""
import sqlite3, argparse

DB_PATH = 'experiments/experiments.db'

parser = argparse.ArgumentParser()
parser.add_argument('--commit', action='store_true')
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Strategy 1: all pieces_placed <= 2 (impossible on valid board)
# Strategy 2: max_height=20 AND holes > 50 (symptom of pre-filled board — even
#             a legitimately full board wouldn't have holes=96 in 1 piece on L7)
removed_short = conn.execute(
    'SELECT COUNT(*) FROM episodes WHERE pieces_placed <= 2'
).fetchone()[0]

removed_corrupt = conn.execute(
    'SELECT COUNT(*) FROM episodes WHERE max_height >= 20 AND holes > 50 AND pieces_placed <= 5'
).fetchone()[0]

print(f'{"DRY RUN" if not args.commit else "DELETING"}: Corrupt episodes:')
print(f'  pieces_placed <= 2:                     {removed_short}')
print(f'  max_height>=20 + holes>50 + pieces<=5:  {removed_corrupt}')

# Preview by level
rows = conn.execute(
    'SELECT level, pieces_placed, COUNT(*) as n FROM episodes '
    'WHERE pieces_placed <= 2 OR (max_height >= 20 AND holes > 50 AND pieces_placed <= 5) '
    'GROUP BY level, pieces_placed ORDER BY level, pieces_placed'
).fetchall()
for r in rows:
    print(f'    L{r["level"]}  pieces={r["pieces_placed"]}  n={r["n"]}')

total_before = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'\nDB before: {total_before}')

if not args.commit:
    print('\nDry run. Re-run with --commit to delete.')
    conn.close()
    exit(0)

conn.execute('DELETE FROM episodes WHERE pieces_placed <= 2')
conn.execute(
    'DELETE FROM episodes WHERE max_height >= 20 AND holes > 50 AND pieces_placed <= 5'
)
conn.commit()

total_after = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'DB after:  {total_after}  (deleted {total_before - total_after})')
conn.close()
print('\nDone. Run generate_tetra_brief.py to refresh.')
