import json, pathlib
f = list(pathlib.Path("experiments/save_states/reward_ram_log/ALE_MontezumaRevenge_v5").glob("*.jsonl"))[0]
data = json.loads(f.read_text().splitlines()[0])
print("Reward +100 at step", data["step"])
print()
print("FROM 0 -> X  (key flag candidates):")
for b in data["changed_bytes"]:
    if b["before"] == 0:
        print(f"  RAM[{b['idx']:3d}]: 0 -> {b['after']:3d}  (0x{b['after']:02X})")
print()
print("ALL changes:")
for b in data["changed_bytes"]:
    note = ""
    if b["before"] == 0: note = " <-- FROM ZERO"
    if b["after"] == 255: note = " <-- TO 0xFF"
    print(f"  RAM[{b['idx']:3d}]: {b['before']:3d} -> {b['after']:3d}  d={b['after']-b['before']:+4d}{note}")
