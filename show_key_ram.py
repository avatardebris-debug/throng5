"""Show the key-collection RAM snapshot and player position from the latest human session."""
import json, glob, os

reward_dir = "experiments/save_states/reward_ram_log/ALE_MontezumaRevenge_v5"
for f in sorted(glob.glob(reward_dir + "/*_rewards.jsonl")):
    events = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    key_events = [e for e in events if e.get("reward") == 100]
    if not key_events:
        continue
    ev = key_events[-1]
    rb = ev["ram_before"]
    ra = ev["ram_after"]
    print(f"Session:  {os.path.basename(f)}")
    print(f"  Key at step {ev['step']}  reward={ev['reward']}")
    print(f"  player x  = RAM[42] = {ra[42]}")
    print(f"  player y  = RAM[43] = {ra[43]}")
    print(f"  room      = RAM[3]  = {ra[3]}")
    print(f"  byte 56   = {ra[56]}  (key_flag: should be 255)")
    print(f"  byte 65   = {ra[65]}  (key_flag2: should be 2)")
    changed = [(i, rb[i], ra[i]) for i in range(128) if rb[i] != ra[i]]
    print(f"  {len(changed)} bytes changed at key pickup:")
    for idx, vb, va in changed:
        label = {3:"room", 42:"player_x", 43:"player_y", 56:"key_flag", 58:"lives", 65:"key_flag2"}.get(idx,"")
        print(f"    RAM[{idx:3d}] {vb:3d} -> {va:3d}  {label}")
