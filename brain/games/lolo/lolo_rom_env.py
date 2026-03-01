"""
lolo_rom_env.py — Gymnasium wrapper for Adventures of Lolo NES ROM via stable-retro.

Bridges the real NES game into the same interface as LoloSimulator,
enabling the brain to play the actual ROM using the same 84-dim
compressed state encoder.

Requirements (Linux only):
  pip install stable-retro
  # Import ROM: python -m retro.import /path/to/roms/

Usage:
  env = LoloROMEnv("Adventures of Lolo (USA)")
  obs = env.reset()
  obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import retro
    HAS_RETRO = True
except ImportError:
    HAS_RETRO = False


# ── Lolo NES RAM Map ──────────────────────────────────────────────────
# Known RAM addresses for Adventures of Lolo (USA)
# These map NES RAM → the format encode_from_ram() expects.
#
# NOTE: These addresses need verification by playing the ROM and
# observing RAM changes. Starting points from NES game analysis:

RAM_MAP = {
    "player_x_pixel": 0x0070,     # Player X position (pixels)
    "player_y_pixel": 0x0050,     # Player Y position (pixels)
    "player_facing":  0x0098,     # Direction facing (0-3)
    "hearts_collected": 0x0062,   # Hearts picked up
    "hearts_total":     0x0063,   # Total hearts in room
    "magic_shots":      0x0064,   # Magic shot count
    "room_number":      0x0040,   # Current room/floor
    "alive":            0x006A,   # Player alive flag
    "chest_open":       0x0065,   # Chest opened flag

    # Grid data starts around 0x0400 in RAM
    "grid_base":        0x0400,   # 13 rows × 11 cols
    "enemy_base":       0x0300,   # Enemy data (8 enemies × 8 bytes)
}

# Tile type mapping: NES RAM value → simulator tile category
NES_TILE_MAP = {
    0x00: 0,   # Empty → walkable
    0x01: 1,   # Rock → blocked
    0x02: 1,   # Tree → blocked
    0x03: 1,   # Water → blocked
    0x04: 0,   # Bridge → walkable
    0x05: 0,   # Desert → walkable
    0x06: 2,   # Heart → collectible
    0x07: 3,   # Chest → goal
    0x08: 4,   # Exit → goal
    0x09: 1,   # Lava → blocked
    0x0A: 5,   # Pushable block → pushable
}

# Action mapping: our 6 actions → NES button combos
# stable-retro uses button arrays: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
# For NES: [B, _, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
ACTION_MAP = {
    0: np.array([0,0,0,0,1,0,0,0,0], dtype=np.int8),  # UP
    1: np.array([0,0,0,0,0,1,0,0,0], dtype=np.int8),  # DOWN
    2: np.array([0,0,0,0,0,0,1,0,0], dtype=np.int8),  # LEFT
    3: np.array([0,0,0,0,0,0,0,1,0], dtype=np.int8),  # RIGHT
    4: np.array([0,0,0,0,0,0,0,0,1], dtype=np.int8),  # SHOOT (A button)
    5: np.array([0,0,0,0,0,0,0,0,0], dtype=np.int8),  # WAIT (no buttons)
}


class LoloROMEnv:
    """
    Gymnasium-style wrapper for the real Lolo ROM.

    Provides the same interface as LoloSimulator:
      - reset() → obs
      - step(action) → obs, reward, done, truncated, info
      - Properties: hearts_collected, hearts_total, won, player_row, player_col
      - save() / load() for state management
    """

    def __init__(
        self,
        game: str = "AdventuresOfLolo-Nes",
        state: str = retro.State.DEFAULT if HAS_RETRO else None,
        render: bool = False,
    ):
        if not HAS_RETRO:
            raise ImportError(
                "stable-retro not installed. "
                "Install with: pip install stable-retro\n"
                "Import ROM: python -m retro.import /path/to/roms/"
            )

        self._game = game
        self._render = render
        self._env = retro.make(
            game=game,
            state=retro.State.NONE,
            inttype=retro.data.Integrations.STABLE,
            use_restricted_actions=retro.Actions.ALL,
        )

        # State tracking
        self._ram = None
        self._prev_hearts = 0
        self._total_reward = 0.0
        self._steps = 0
        self._won = False
        self._alive = True

        # Properties matching LoloSimulator API
        self.hearts_collected = 0
        self.hearts_total = 0
        self.player_row = 0
        self.player_col = 0

    def reset(self, **kwargs) -> np.ndarray:
        """Reset the environment. Returns pixel observation."""
        obs, info = self._env.reset(**kwargs)
        self._ram = self._env.get_ram()
        self._prev_hearts = 0
        self._total_reward = 0.0
        self._steps = 0
        self._won = False
        self._alive = True
        self._update_state_from_ram()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.

        Args:
            action: 0-5 (UP, DOWN, LEFT, RIGHT, SHOOT, WAIT)

        Returns:
            (obs, reward, done, truncated, info)
        """
        # Convert our action to NES buttons
        buttons = ACTION_MAP.get(action, ACTION_MAP[5])

        # Step the emulator (hold button for a few frames for responsiveness)
        total_reward = 0.0
        obs = None
        done = False
        for _ in range(4):  # 4-frame action repeat
            obs, reward, terminated, truncated, info = self._env.step(buttons)
            total_reward += reward
            if terminated or truncated:
                done = True
                break

        self._ram = self._env.get_ram()
        self._steps += 1
        self._update_state_from_ram()

        # Custom reward shaping
        shaped_reward = self._shape_reward(total_reward)
        self._total_reward += shaped_reward

        # Check win/death
        if self.hearts_collected >= self.hearts_total and self.hearts_total > 0:
            self._won = True

        info = {
            "hearts": self.hearts_collected,
            "hearts_total": self.hearts_total,
            "pos": (self.player_row, self.player_col),
            "won": self._won,
            "steps": self._steps,
        }

        return obs, shaped_reward, done, truncated, info

    def _update_state_from_ram(self):
        """Read RAM and update state properties."""
        if self._ram is None:
            return

        ram = self._ram

        # Player position (pixel → grid cell)
        px = int(ram[RAM_MAP["player_x_pixel"]])
        py = int(ram[RAM_MAP["player_y_pixel"]])
        self.player_row = max(0, min(12, py // 16))
        self.player_col = max(0, min(10, px // 16))

        # Hearts
        self.hearts_collected = int(ram[RAM_MAP["hearts_collected"]])
        self.hearts_total = int(ram[RAM_MAP["hearts_total"]])

        # Alive
        self._alive = bool(ram[RAM_MAP["alive"]])

    def _shape_reward(self, base_reward: float) -> float:
        """Shape reward to match simulator training."""
        reward = 0.0

        # Heart collected
        if self.hearts_collected > self._prev_hearts:
            reward += 10.0
            self._prev_hearts = self.hearts_collected

        # Game reward (score-based from emulator)
        if base_reward > 0:
            reward += base_reward * 0.01  # Scale down emulator rewards

        # Death penalty
        if not self._alive:
            reward -= 5.0

        # Win bonus
        if self._won:
            reward += 50.0

        # Small step penalty
        reward -= 0.01

        return reward

    def get_ram_state(self) -> Dict[str, Any]:
        """
        Extract structured state from RAM for encode_from_ram().

        Returns dict matching LoloCompressedState.encode_from_ram() format.
        """
        if self._ram is None:
            return {}

        ram = self._ram

        # Extract grid (13×11)
        grid = np.zeros((13, 11), dtype=np.int32)
        grid_base = RAM_MAP["grid_base"]
        for row in range(13):
            for col in range(11):
                addr = grid_base + row * 11 + col
                if addr < len(ram):
                    tile_val = int(ram[addr])
                    grid[row, col] = NES_TILE_MAP.get(tile_val, 0)

        # Extract enemies
        enemies = []
        enemy_base = RAM_MAP["enemy_base"]
        for i in range(8):
            base = enemy_base + i * 8
            if base + 7 < len(ram):
                ex = int(ram[base])
                ey = int(ram[base + 1])
                etype = int(ram[base + 2])
                alive = bool(ram[base + 3])
                if alive and (ex > 0 or ey > 0):
                    enemies.append((
                        ey // 16,      # row
                        ex // 16,      # col
                        0,             # facing
                        etype,         # type
                        alive,         # alive
                        False,         # is_egg
                        etype >= 8,    # dangerous (rough heuristic)
                    ))

        return {
            "grid": grid,
            "player_row": self.player_row,
            "player_col": self.player_col,
            "hearts_collected": self.hearts_collected,
            "hearts_total": self.hearts_total,
            "chest_open": bool(ram[RAM_MAP["chest_open"]]) if RAM_MAP["chest_open"] < len(ram) else False,
            "has_jewel": False,
            "magic_shots": int(ram[RAM_MAP["magic_shots"]]) if RAM_MAP["magic_shots"] < len(ram) else 0,
            "step_count": self._steps,
            "enemies": enemies,
        }

    def save(self) -> bytes:
        """Save emulator state."""
        return self._env.em.get_state()

    def load(self, state: bytes):
        """Load emulator state."""
        self._env.em.set_state(state)
        self._ram = self._env.get_ram()
        self._update_state_from_ram()

    @property
    def won(self) -> bool:
        return self._won

    def close(self):
        self._env.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


def test_rom():
    """Quick test: load ROM and take a few steps."""
    print("Testing Lolo ROM environment...")

    from brain.games.lolo.lolo_compressed_state import LoloCompressedState
    encoder = LoloCompressedState()

    env = LoloROMEnv()
    obs = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Hearts: {env.hearts_collected}/{env.hearts_total}")
    print(f"  Player: ({env.player_row}, {env.player_col})")

    # Take some steps
    for action in [0, 3, 3, 1, 1, 4]:  # UP, RIGHT, RIGHT, DOWN, DOWN, SHOOT
        obs, reward, done, truncated, info = env.step(action)
        ram_state = env.get_ram_state()
        compressed = encoder.encode_from_ram(ram_state)
        print(f"  Action {action}: reward={reward:.2f}, "
              f"hearts={info['hearts']}/{info['hearts_total']}, "
              f"compressed={compressed[:5]}...")

    env.close()
    print("  ROM test complete!")


if __name__ == "__main__":
    test_rom()
