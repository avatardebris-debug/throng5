import json

baseline = json.load(open('curriculum_baseline_2_to_7.json'))
dreamer = json.load(open('curriculum_2_to_7.json'))
tetra = json.load(open('curriculum_with_tetra_real.json'))

print('FULL COMPARISON: Baseline vs Dreamer vs Dreamer+Tetra')
print('='*80)
print()
print(f'{"Level":<8} {"Baseline":<10} {"Dreamer":<10} {"+Tetra":<10} {"D-B":<8} {"T-D":<8} {"T-B":<8}')
print('-'*80)

for b, d, t in zip(baseline, dreamer, tetra):
    d_vs_b = d['mean_lines'] - b['mean_lines']
    t_vs_d = t['mean_lines'] - d['mean_lines']
    t_vs_b = t['mean_lines'] - b['mean_lines']
    
    print(f'{b["level"]:<8} {b["mean_lines"]:<10.2f} {d["mean_lines"]:<10.2f} {t["mean_lines"]:<10.2f} {d_vs_b:<+8.2f} {t_vs_d:<+8.2f} {t_vs_b:<+8.2f}')

print()
print('Legend:')
print('  D-B = Dreamer improvement over Baseline')
print('  T-D = Tetra improvement over Dreamer-only')
print('  T-B = Total improvement (Dreamer+Tetra over Baseline)')
