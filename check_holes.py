import sqlite3
conn = sqlite3.connect('experiments/experiments.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    'SELECT holes, max_height, bumpiness, lines_cleared FROM episodes ORDER BY timestamp DESC LIMIT 10'
).fetchall()
print('Last 10 episodes — holes / max_height / bumpiness / lines:')
for row in rows:
    print(f"  holes={row['holes']:3}  height={row['max_height']:2}  "
          f"bump={row['bumpiness']:5.1f}  lines={row['lines_cleared']}")

r2 = conn.execute(
    'SELECT AVG(holes), MIN(holes), MAX(holes) FROM episodes WHERE timestamp > unixepoch() - 300'
).fetchone()
print(f'Last 5 min: avg_holes={r2[0]:.1f}  min={r2[1]}  max={r2[2]}')
conn.close()
