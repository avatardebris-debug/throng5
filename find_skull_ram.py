"""
find_skull_ram.py
=================
Identifies the RAM byte(s) that track the skull's x position.

Method:
  1. Load the lower_floor_reached frontier checkpoint (skull is visible)
  2. Hold NOOP for ~200 steps while skull oscillates left<->right
  3. Find bytes that: change multiple times, stay in 0-160 range, oscillate
     (value increases then decreases like a bouncing patrol)

Run:
    python find_skull_ram.py
"""

import pickle
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from pathlib import Path

SAVE_DIR = Path("experiments/save_states/ALE_MontezumaRevenge_v5")
NOOP     = 0
N_STEPS  = 300

# ── Find best (deepest) frontier file ─────────────────────────────────
candidates = sorted(SAVE_DIR.glob("frontier_lower_floor*.bin"))
if not candidates:
    candidates = sorted(SAVE_DIR.glob("frontier_*.bin"))
if not candidates:
    raise FileNotFoundError(f"No frontier .bin files in {SAVE_DIR}")
frontier = candidates[-1]
print(f"Loading: {frontier.name}")

env = gym.make("ALE/MontezumaRevenge-v5", obs_type="ram",
               render_mode=None, frameskip=1)
env.reset()

with open(frontier, "rb") as f:
    saved = pickle.load(f)
env.unwrapped.ale.restoreState(saved["ram"])

# Warmup: let agent land from any mid-air checkpoint position
for _ in range(80):
    env.step(NOOP)

# ── Collect RAM history ────────────────────────────────────────────────
history = []
for _ in range(N_STEPS):
    obs, _, term, trunc, _ = env.step(NOOP)
    if term or trunc:
        break
    history.append(np.array(obs, dtype=np.int16))

env.close()

ram_arr = np.stack(history)  # shape (N_STEPS, 128)

# ── Analysis ───────────────────────────────────────────────────────────
print(f"\nAnalysing {N_STEPS} steps of RAM...\n")
print(f"{'Byte':>4}  {'min':>4}  {'max':>4}  {'range':>5}  {'changes':>7}  "
      f"{'direction_flips':>14}  note")
print("-" * 70)

results = []
for byte_idx in range(128):
    vals = ram_arr[:, byte_idx].tolist()
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0:
        continue  # static byte
    # Count direction changes (proxy for oscillation)
    changes = sum(1 for i in range(1, len(vals)) if vals[i] != vals[i-1])
    diffs = [vals[i] - vals[i-1] for i in range(1, len(vals)) if vals[i] != vals[i-1]]
    flips = sum(1 for i in range(1, len(diffs))
                if diffs[i] != 0 and diffs[i-1] != 0
                and (diffs[i] > 0) != (diffs[i-1] > 0))
    results.append((byte_idx, mn, mx, rng, changes, flips, vals))

# Sort by direction flips * range (most oscillatory AND wide range first)
results.sort(key=lambda r: r[5] * r[3], reverse=True)

# Print top 15 candidates
for byte_idx, mn, mx, rng, changes, flips, vals in results[:15]:
    note = ""
    if 0 <= mn and mx <= 160:
        note = "<-- x-range candidate"
    if byte_idx in (42, 43):
        note = "<-- PLAYER x/y (known)"
    print(f"  {byte_idx:3d}  {mn:4d}  {mx:4d}  {rng:5d}  {changes:7d}  "
          f"{flips:14d}  {note}")

print("\nTop candidate for skull x:", results[0][0] if results else "none")
print("(should have range 0-160, many direction flips = oscillating patrol)")
