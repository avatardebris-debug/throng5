import sys, sqlite3
sys.path.insert(0, '.')
con = sqlite3.connect('experiments/experiments.db')
con.row_factory = sqlite3.Row

dups = con.execute('''
    SELECT name, COUNT(*) as cnt,
           GROUP_CONCAT(status) as statuses,
           GROUP_CONCAT(evidence_count) as evidence,
           GROUP_CONCAT(game) as games
    FROM hypotheses
    GROUP BY name HAVING cnt > 1
    ORDER BY cnt DESC LIMIT 15
''').fetchall()

print(f'Duplicate hypothesis names: {len(dups)}')
for r in dups:
    print(f'  [{r["cnt"]}x] {r["name"]}')
    print(f'        statuses={r["statuses"]}  evidence={r["evidence"]}  games={r["games"]}')

total = con.execute('SELECT COUNT(*) FROM hypotheses').fetchone()[0]
candidates = con.execute("SELECT COUNT(*) FROM hypotheses WHERE evidence_count < 10").fetchone()[0]
print(f'\nTotal rows: {total}  Untested candidates (evidence<10): {candidates}')

# Show confidence vs win_rate divergence
print('\nConfidence vs win_rate sample (active hypotheses):')
rows = con.execute('''
    SELECT name, confidence, win_rate, evidence_count
    FROM hypotheses WHERE evidence_count > 20
    ORDER BY ABS(confidence - win_rate) DESC LIMIT 8
''').fetchall()
for r in rows:
    print(f'  {r["name"]:<35} conf={r["confidence"]:.2f}  win_rate={r["win_rate"]:.2f}  n={r["evidence_count"]}')
con.close()
