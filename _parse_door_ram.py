import json
from pathlib import Path

log_dir = Path("experiments/save_states/reward_ram_log/ALE_MontezumaRevenge_v5")
files = sorted(log_dir.glob("*_rewards.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)

for f in files[:3]:
    print(f"=== {f.name} ===")
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    found = False
    for line in lines:
        ev = json.loads(line)
        if ev.get("reward", 0) >= 100:
            found = True
            print(f"  step={ev['step']}  reward={ev['reward']}")
            rb = ev.get("ram_before", [])
            ra = ev.get("ram_after",  [])
            if rb and ra:
                changed = [(i, rb[i], ra[i]) for i in range(min(len(rb), len(ra))) if rb[i] != ra[i]]
                print(f"  Changed RAM bytes ({len(changed)} total):")
                for idx, bef, aft in changed:
                    print(f"    RAM[{idx:3d}]  {bef:3d} -> {aft:3d}")
    if not found:
        print("  (no reward >= 100 in this log)")
    print()
