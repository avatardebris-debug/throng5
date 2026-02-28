import json
from pathlib import Path

session_id = "ses_1993e803-536a-48a4-b981-e25dc2eb40a6"
f = Path(f"experiments/atari_events/ALE_MontezumaRevenge_v5/{session_id}.jsonl")
evs = [json.loads(l) for l in f.read_text(encoding="utf-8").strip().splitlines()]

print(f"Session: {len(evs)} steps")

# Track RAM[3] (room), RAM[42] (x), RAM[43] (y) across session
prev_room = None
for e in evs:
    ram = e.get("ram")
    if not ram or len(ram) < 44:
        continue
    room = ram[3]
    x, y = ram[42], ram[43]
    if room != prev_room:
        print(f"  step={e['step']:5d}  room={room}  x={x}  y={y}  reward={e.get('reward',0)}")
        # Show all changed bytes on room transition
        if prev_room is not None:
            idx = next(i for i, ev in enumerate(evs) if ev.get("step") == e["step"])
            if idx > 0:
                rb = evs[idx-1].get("ram")
                if rb:
                    changed = [(i, rb[i], ram[i]) for i in range(128) if rb[i] != ram[i]]
                    print(f"    Changed bytes ({len(changed)}): {[(i,b,a) for i,b,a in changed[:15]]}")
        prev_room = room

# Summary
rooms = sorted(set(e["ram"][3] for e in evs if e.get("ram") and len(e["ram"]) >= 4))
print(f"\nRooms visited: {rooms}")
print(f"Save state was at step 402, session ended at step {evs[-1]['step']}")
