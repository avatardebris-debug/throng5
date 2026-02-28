"""Try loading each NES ROM to find which ones work."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roms", "nes")
roms = sorted([f for f in os.listdir(ROM_DIR) if f.endswith(".nes")])

print(f"Found {len(roms)} ROMs in {ROM_DIR}\n")

working = []
for rom_name in roms:
    rom_path = os.path.join(ROM_DIR, rom_name)
    try:
        from nes_py.nes_env import NESEnv
        env = NESEnv(rom_path)
        obs = env.reset()
        # Run 10 quick steps
        for _ in range(10):
            obs, r, done, info = env.step(env.action_space.sample())
            if done:
                obs = env.reset()
        env.close()
        print(f"  OK  {rom_name} (obs={obs.shape}, actions={env.action_space.n})")
        working.append(rom_name)
    except Exception as e:
        err = str(e)[:60]
        print(f"  ERR {rom_name}: {err}")

print(f"\n{len(working)}/{len(roms)} ROMs loaded successfully")
if working:
    print(f"Working ROMs: {working}")
