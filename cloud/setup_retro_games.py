"""
setup_retro_games.py — Register custom game ROMs with stable-retro.

Run once on any new machine:
  python3 cloud/setup_retro_games.py
"""

import os
import shutil
import sys

try:
    import retro
    RETRO_DATA = retro.data.path()
except ImportError:
    print("ERROR: stable-retro not installed. pip install stable-retro")
    sys.exit(1)

CUSTOM_GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "retro_games")
ROMS_DIR = os.path.join(os.path.dirname(__file__), "..", "roms", "nes")

ROM_MAP = {
    "Adventures of Lolo (USA).nes": "AdventuresOfLolo-Nes",
    "Adventures of Lolo 2 (USA).nes": "AdventuresOfLolo2-Nes",
    "Adventures of Lolo 3 (USA).nes": "AdventuresOfLolo3-Nes",
}

def setup_game(game_name, rom_filename):
    game_def_dir = os.path.join(CUSTOM_GAMES_DIR, game_name)
    rom_path = os.path.join(ROMS_DIR, rom_filename)
    dest_dir = os.path.join(RETRO_DATA, "stable", game_name)

    if not os.path.isdir(game_def_dir):
        print(f"  ⚠ No game definition for {game_name}")
        return False
    if not os.path.exists(rom_path):
        print(f"  ⚠ ROM not found: {rom_path}")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(game_def_dir):
        shutil.copy2(os.path.join(game_def_dir, fname), os.path.join(dest_dir, fname))
    shutil.copy2(rom_path, os.path.join(dest_dir, "rom.nes"))

    print(f"  ✅ {game_name}")
    return True

def main():
    print("Setting up custom games for stable-retro")
    print(f"  Retro data: {RETRO_DATA}")

    registered = 0
    for rom_file, game_name in ROM_MAP.items():
        if setup_game(game_name, rom_file):
            registered += 1

    print(f"\n  Registered {registered}/{len(ROM_MAP)} games")

    # Verify
    for game_name in ROM_MAP.values():
        try:
            env = retro.make(game=game_name)
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            print(f"  ✅ {game_name}: obs={obs.shape}")
            env.close()
        except Exception as e:
            print(f"  ❌ {game_name}: {e}")

if __name__ == "__main__":
    main()
