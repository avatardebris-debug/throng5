"""
load_rom.py — Simple script to load and play any NES/SNES/Genesis ROM.

Usage:
    # List available ROMs:
    python load_rom.py --list

    # Load a specific ROM file:
    python load_rom.py roms/nes/SuperMarioBros.nes

    # Load by name (auto-finds in roms/ folder):
    python load_rom.py mario

    # Import ROMs into gym-retro (one-time setup):
    python load_rom.py --import-all
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.config import NES_ROMS_DIR, SNES_ROMS_DIR, GENESIS_ROMS_DIR, ROMS_DIR


ROM_EXTENSIONS = {
    ".nes": "nes",
    ".sfc": "snes",
    ".smc": "snes",
    ".md": "genesis",
    ".bin": "genesis",
    ".gen": "genesis",
}


def find_all_roms():
    """Scan roms/ directory tree for all ROM files."""
    roms = []
    for ext, platform in ROM_EXTENSIONS.items():
        for f in ROMS_DIR.rglob(f"*{ext}"):
            roms.append({"path": f, "name": f.stem, "platform": platform, "ext": ext})
    return roms


def find_rom(query: str):
    """Find a ROM by partial name match."""
    query_lower = query.lower()
    roms = find_all_roms()
    # Exact match first
    for rom in roms:
        if rom["name"].lower() == query_lower:
            return rom
    # Partial match
    for rom in roms:
        if query_lower in rom["name"].lower():
            return rom
    return None


def list_roms():
    """Print all discovered ROMs."""
    roms = find_all_roms()
    if not roms:
        print(f"No ROMs found in {ROMS_DIR}")
        print(f"\nDrop your ROM files into:")
        print(f"  NES:     {NES_ROMS_DIR}")
        print(f"  SNES:    {SNES_ROMS_DIR}")
        print(f"  Genesis: {GENESIS_ROMS_DIR}")
        return

    print(f"Found {len(roms)} ROM(s):\n")
    for rom in roms:
        print(f"  [{rom['platform']:7s}] {rom['name']:30s} {rom['path']}")


def import_to_retro():
    """Import all ROMs in roms/ into gym-retro."""
    try:
        import retro
    except ImportError:
        print("gym-retro not installed. Run: pip install gym-retro")
        return

    for platform_dir in [NES_ROMS_DIR, SNES_ROMS_DIR, GENESIS_ROMS_DIR]:
        if platform_dir.exists() and any(platform_dir.iterdir()):
            print(f"Importing ROMs from {platform_dir}...")
            try:
                retro.data.merge(str(platform_dir))
                print(f"  Done.")
            except Exception as e:
                print(f"  Error: {e}")
                print(f"  Try manually: python -m retro.import {platform_dir}")


def load_rom(rom_path: str):
    """Load a ROM and create environment + adapter."""
    path = Path(rom_path)

    if not path.exists():
        # Try finding by name
        rom = find_rom(rom_path)
        if rom:
            path = rom["path"]
        else:
            print(f"ROM not found: {rom_path}")
            print("Use --list to see available ROMs")
            return None, None

    print(f"Loading: {path.name} ({path})")

    # Try gym-retro first
    try:
        import retro
        game_name = path.stem
        env = retro.make(game=game_name)
        print(f"Loaded via gym-retro as '{game_name}'")

        from brain.environments.rom_adapter_factory import ROMAdapterFactory
        factory = ROMAdapterFactory()
        adapter, fingerprint = factory.create_from_env(env, game_name)
        print(f"Environment fingerprint:")
        print(f"  Platform:     {fingerprint.platform}")
        print(f"  Actions:      {fingerprint.n_actions} ({fingerprint.action_space_type})")
        print(f"  Obs shape:    {fingerprint.obs_shape}")
        print(f"  Reward:       {fingerprint.reward_density}")
        return env, adapter

    except ImportError:
        print("gym-retro not installed. Run: pip install gym-retro")
        print("For now, you can still use the adapter with manual observation passing.")
        return None, None
    except Exception as e:
        print(f"Could not load via retro: {e}")
        print("Make sure the ROM is imported: python load_rom.py --import-all")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load NES/SNES/Genesis ROMs")
    parser.add_argument("rom", nargs="?", help="ROM file path or name to search for")
    parser.add_argument("--list", action="store_true", help="List all available ROMs")
    parser.add_argument("--import-all", action="store_true", help="Import all ROMs into gym-retro")
    args = parser.parse_args()

    if args.list:
        list_roms()
    elif args.import_all:
        import_to_retro()
    elif args.rom:
        env, adapter = load_rom(args.rom)
        if env:
            print("\nReady! Use the brain to play:")
            print("  from brain.orchestrator import WholeBrain")
            print("  brain = WholeBrain(n_features=84, n_actions=fingerprint.n_actions)")
            print("  brain.set_adapter(adapter)")
            env.close()
    else:
        list_roms()
