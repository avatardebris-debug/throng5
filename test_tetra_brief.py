import json
from throng4.storage import ExperimentDB

db = ExperimentDB('experiments/experiments.db')
brief = db.generate_tetra_brief(game='tetris', last_n_episodes=500)

print('=== GAME CONTEXT ===')
ctx = brief['game_context']
print(f"  Total episodes: {ctx['total_episodes']}")
print(f"  Training span: {ctx['training_span_days']} days")
for lv in ctx['levels_trained']:
    print(f"  L{lv['level']}: {lv['n']} eps, avg={lv['avg']:.1f}, best={lv['best']}")

print()
print('=== HYPOTHESIS LEDGER ===')
hl = brief['hypothesis_ledger']
print(f"  Active: {len(hl['active'])}")
print(f"  Retired: {len(hl['retired'])}")
print(f"  Untested: {len(hl['untested_candidates'])}")
print(f"  Key insight: {hl['key_insight']}")

print()
print('=== FAILURE vs SUCCESS PATTERNS ===')
fp = brief['failure_patterns']
sp = brief['success_patterns']
print(f"  Failures  (n={fp['sample_size']}): {fp.get('interpretation','')}")
print(f"  Successes (n={sp['sample_size']}): {sp.get('interpretation','')}")
if 'delta_vs_failure' in sp:
    d = sp['delta_vs_failure']
    print(f"  Deltas: height={d['height_delta']}, holes={d['holes_delta']}, bump={d['bumpiness_delta']}")

print()
print('=== OPEN QUESTIONS ===')
for q in brief['open_questions']:
    print(f"  - {q}")

print()
print('=== RECOMMENDED FOCUS ===')
for r in brief['recommended_focus']:
    print(f"  - {r}")

print()
print('=== PROMPT FOR TETRA ===')
print(brief['prompt_for_tetra'])

# Also save the full brief to a file for inspection
with open('experiments/tetra_brief.json', 'w') as f:
    json.dump(brief, f, indent=2, default=str)
print('\nFull brief saved to experiments/tetra_brief.json')
db.close()
