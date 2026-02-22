import json

h = json.loads(open("benchmark_results/human_ALE_MontezumaRevenge_v5.json").read())
print("Human-seeded Montezuma:")
print(f"  mean={h['mean_reward']:.1f}  max={h['max_reward']:.1f}  seeded={h['n_seeded']}  warmup={h['n_warmup_steps']}")

bs = json.loads(open("benchmark_results/baseline_summary.json").read())
print(f"\nBaseline: {len(bs)} games")
for g in sorted(bs, key=lambda x: x["mean_reward"], reverse=True):
    print(f"  {g['game'].replace('ALE/',''):<28} mean={g['mean_reward']:9.1f}  max={g['max_reward']:9.1f}")

hs = json.loads(open("benchmark_results/human_summary.json").read())
print(f"\nHuman-seeded summary: {len(hs)} games")
for g in hs:
    print(f"  {g['game'].replace('ALE/',''):<28} mean={g['mean_reward']:9.1f}  max={g['max_reward']:9.1f}  seeded={g['n_seeded']}")
