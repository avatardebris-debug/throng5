import sqlite3, time

conn = sqlite3.connect('experiments/experiments.db')
conn.row_factory = sqlite3.Row

today = time.mktime(time.strptime('2026-02-19 00:00:00', '%Y-%m-%d %H:%M:%S'))

print('Remaining short episodes (all time):')
rows = conn.execute(
    'SELECT level, pieces_placed, COUNT(*) as n '
    'FROM episodes WHERE pieces_placed <= 2 '
    'GROUP BY level, pieces_placed ORDER BY level, pieces_placed'
).fetchall()
for r in rows:
    print(f'  L{r["level"]}  pieces={r["pieces_placed"]}  n={r["n"]}')
total = sum(r['n'] for r in rows)
print(f'  TOTAL: {total}')

# Check the v25 1-piece deaths - are they actually suspicious?
print('\nSample v25 1-piece L7 deaths (board state):')
rows2 = conn.execute(
    'SELECT pieces_placed, max_height, holes, bumpiness, score '
    'FROM episodes WHERE level=7 AND pieces_placed <= 2 AND policy_pack_version=25 '
    'LIMIT 10'
).fetchall()
for r in rows2:
    print(f'  pieces={r["pieces_placed"]}  max_h={r["max_height"]}  '
          f'holes={r["holes"]}  score={r["score"]:.2f}')

conn.close()
