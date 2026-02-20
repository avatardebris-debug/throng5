"""
purge_corrupt_board_episodes.py

Removes episodes from the pre-v20 sessions today that had board corruption
(max_height=20 + holes~96 after 1 piece = entire board filled by some bug
in the env before pack v20 was used tonight).

These are NOT valid training data.

Run: python purge_corrupt_board_episodes.py           # dry-run
     python purge_corrupt_board_episodes.py --commit  # delete
"""
import sqlite3, time, argparse

today = time.mktime(time.strptime('2026-02-19 00:00:00', '%Y-%m-%d %H:%M:%S'))
CORRUPT_MAX_PACK = 19  # packs v14-v16 were from the buggy earlier session

DB_PATH = 'experiments/experiments.db'

parser = argparse.ArgumentParser()
parser.add_argument('--commit', action='store_true')
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Show what we'd remove: short episodes from pre-v20 packs today
rows = conn.execute(
    'SELECT policy_pack_version, level, pieces_placed, COUNT(*) as n '
    'FROM episodes '
    'WHERE pieces_placed <= 2 AND timestamp >= ? AND policy_pack_version <= ? '
    'GROUP BY policy_pack_version, level, pieces_placed '
    'ORDER BY policy_pack_version, level',
    (today, CORRUPT_MAX_PACK)
).fetchall()

total = sum(r['n'] for r in rows)
print(f'{"DRY RUN — " if not args.commit else ""}Corrupt-board episodes (pack<=v{CORRUPT_MAX_PACK}, today):')
for r in rows:
    print(f'  pack=v{r["policy_pack_version"]}  L{r["level"]}  pieces={r["pieces_placed"]}  n={r["n"]}')
print(f'  TOTAL: {total}')

total_before = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'\nDB before: {total_before} total episodes')

if not args.commit:
    print('\nDry run complete. Re-run with --commit to delete.')
    conn.close()
    exit(0)

conn.execute(
    'DELETE FROM episodes '
    'WHERE pieces_placed <= 2 AND timestamp >= ? AND policy_pack_version <= ?',
    (today, CORRUPT_MAX_PACK)
)
conn.commit()

total_after = conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
print(f'DB after:  {total_after} total episodes')
print(f'Deleted:   {total_before - total_after} rows')
conn.close()
print('\nDone. Run generate_tetra_brief.py to refresh.')
