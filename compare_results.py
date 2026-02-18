import json

dreamer = json.load(open('curriculum_2_to_7.json'))
baseline = json.load(open('curriculum_baseline_2_to_7.json'))

print('BASELINE vs DREAMER COMPARISON')
print('='*80)
print('\nLevel | Baseline Mean | Dreamer Mean | Delta  | Improvement | Base Max | Dream Max')
print('------|---------------|--------------|--------|-------------|----------|----------')

total_baseline = 0
total_dreamer = 0

for b, d in zip(baseline, dreamer):
    level = b['level']
    b_mean = b['mean_lines']
    d_mean = d['mean_lines']
    delta = d_mean - b_mean
    pct = (delta / b_mean * 100) if b_mean > 0 else 0
    
    total_baseline += b_mean
    total_dreamer += d_mean
    
    print(f'{level:5} | {b_mean:13.2f} | {d_mean:12.2f} | {delta:+6.2f} | {pct:+10.1f}% | {b["max_lines"]:8} | {d["max_lines"]:9}')

print('------|---------------|--------------|--------|-------------|----------|----------')
avg_baseline = total_baseline / len(baseline)
avg_dreamer = total_dreamer / len(dreamer)
avg_delta = avg_dreamer - avg_baseline
avg_pct = (avg_delta / avg_baseline * 100) if avg_baseline > 0 else 0

print(f'{"AVG":5} | {avg_baseline:13.2f} | {avg_dreamer:12.2f} | {avg_delta:+6.2f} | {avg_pct:+10.1f}% |          |          ')

print('\n' + '='*80)
print('\nKEY INSIGHTS:')
print(f'  - Overall improvement: {avg_pct:+.1f}%')
print(f'  - Dreamer helped most on: Level {max(zip(baseline, dreamer), key=lambda x: x[1]["mean_lines"]-x[0]["mean_lines"])[0]["level"]}')
print(f'  - Total episodes: {sum(s["episodes"] for s in dreamer)}')
print(f'  - Total time (dreamer): {sum(s["elapsed_s"] for s in dreamer):.1f}s')
print(f'  - Total time (baseline): {sum(s["elapsed_s"] for s in baseline):.1f}s')
