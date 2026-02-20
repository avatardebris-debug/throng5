import sqlite3, time

conn = sqlite3.connect('experiments/experiments.db')
conn.row_factory = sqlite3.Row

today = time.mktime(time.strptime('2026-02-19 00:00:00', '%Y-%m-%d %H:%M:%S'))

print('Remaining short episodes (today only):')
rows = conn.execute(
    'SELECT level, pieces_placed, COUNT(*) as n '
    'FROM episodes WHERE pieces_placed <= 2 AND timestamp >= ? '
    'GROUP BY level, pieces_placed ORDER BY level, pieces_placed',
    (today,)
).fetchall()
for r in rows:
    print(f'  L{r["level"]}  pieces={r["pieces_placed"]}  n={r["n"]}')
total = sum(r['n'] for r in rows)
print(f'  TOTAL: {total}')

print()
# For L7 specifically, are they at the START of the session (episode_num small)?
print('L7 pieces<=2 by episode_num bucket (today):')
rows2 = conn.execute(
    'SELECT episode_num, pieces_placed FROM episodes '
    'WHERE level=7 AND pieces_placed <= 2 AND timestamp >= ? '
    'ORDER BY episode_num',
    (today,)
).fetchall()
if rows2:
    # Bucket into 5 groups
    total_l7 = conn.execute(
        'SELECT COUNT(*) as n FROM episodes WHERE level=7 AND timestamp >= ?',
        (today,)
    ).fetchone()['n']
    bucket_size = max(total_l7 // 5, 1)
    print(f'  (total L7 episodes today: {total_l7}, bucket_size={bucket_size})')
    buckets = {}
    for r in rows2:
        b = r['episode_num'] // bucket_size
        buckets[b] = buckets.get(b, 0) + 1
    for b in sorted(buckets):
        print(f'  ep_bucket {b*bucket_size}-{(b+1)*bucket_size}: {buckets[b]} short episodes')

conn.close()
