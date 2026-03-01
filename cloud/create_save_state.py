"""
create_save_state.py — Boot the Lolo ROM and create a save state at Floor 1.

Run once, then all future training/testing loads from the save state instantly.

Usage:
  python3 cloud/create_save_state.py

This auto-presses START through menus, waits for gameplay, then saves.
If the auto-skip doesn't work, increase the frame counts.
"""

import os
import sys
import time

sys.path.insert(0, ".")
import numpy as np

try:
    import retro
except ImportError:
    import stable_retro as retro

GAME = "AdventuresOfLolo-Nes"
SAVE_DIR = os.path.join("retro_games", "AdventuresOfLolo-Nes")
SAVE_PATH = os.path.join(SAVE_DIR, "Start.state")

START = np.array([0,0,0,1,0,0,0,0,0], dtype=np.int8)
A_BTN = np.array([0,0,0,0,0,0,0,0,1], dtype=np.int8)
NONE  = np.array([0,0,0,0,0,0,0,0,0], dtype=np.int8)


def wait_frames(env, n, btn=None):
    """Step N frames with optional button held."""
    if btn is None:
        btn = NONE
    obs = None
    for _ in range(n):
        obs, _, _, _, _ = env.step(btn)
    return obs


def main():
    print("=" * 60)
    print("  Creating Lolo save state at Floor 1")
    print("=" * 60)

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        inttype=retro.data.Integrations.STABLE,
        use_restricted_actions=retro.Actions.ALL,
    )
    env.reset()

    # Phase 1: Wait for intro animation (~5 seconds = 300 frames at 60fps)
    print("  Waiting for intro animation...", flush=True)
    wait_frames(env, 300)

    # Phase 2: Press START to skip title
    print("  Pressing START (title screen)...", flush=True)
    wait_frames(env, 5, START)
    wait_frames(env, 60)

    # Phase 3: Press START again (file/game select)
    print("  Pressing START (game select)...", flush=True)
    wait_frames(env, 5, START)
    wait_frames(env, 60)

    # Phase 4: Press A or START to confirm
    print("  Pressing A (confirm)...", flush=True)
    wait_frames(env, 5, A_BTN)
    wait_frames(env, 60)

    # Phase 5: Another START/A for floor description
    print("  Pressing START (floor description)...", flush=True)
    wait_frames(env, 5, START)
    wait_frames(env, 120)  # Wait for floor text + room load

    # Phase 6: One more START/A if needed
    print("  Pressing A (start room)...", flush=True)
    wait_frames(env, 5, A_BTN)
    wait_frames(env, 120)

    # Phase 7: Press START one more time just in case
    print("  Final START press...", flush=True)
    wait_frames(env, 5, START)
    wait_frames(env, 60)

    # Check RAM to see if we're in gameplay
    ram = env.get_ram()
    print(f"\n  RAM check:")
    print(f"    Addr 0x0062 (hearts_collected?): {ram[0x0062]}")
    print(f"    Addr 0x0063 (hearts_total?):     {ram[0x0063]}")
    print(f"    Addr 0x0040 (room?):             {ram[0x0040]}")
    print(f"    Addr 0x0070 (player_x?):         {ram[0x0070]}")
    print(f"    Addr 0x0050 (player_y?):         {ram[0x0050]}")
    print(f"    Addr 0x006A (alive?):            {ram[0x006A]}")

    # Dump first 256 bytes of RAM for analysis
    print(f"\n  First 256 RAM bytes:")
    for row in range(16):
        addr = row * 16
        hex_vals = " ".join(f"{ram[addr+i]:02X}" for i in range(16))
        print(f"    0x{addr:04X}: {hex_vals}")

    # Save the state
    state = env.em.get_state()
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        f.write(state)

    print(f"\n  ✅ Save state written to: {SAVE_PATH}")
    print(f"     Size: {len(state)} bytes")

    # Also copy to retro's data dir for auto-loading
    retro_dest = os.path.join(retro.data.path(), "stable", GAME, "Start.state")
    with open(retro_dest, "wb") as f:
        f.write(state)
    print(f"  ✅ Copied to retro data: {retro_dest}")

    env.close()
    print(f"\n  Done! Future runs will load from this save state.")


if __name__ == "__main__":
    main()
