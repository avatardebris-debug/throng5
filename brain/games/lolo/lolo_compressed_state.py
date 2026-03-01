"""
lolo_compressed_state.py — Tetris-style compressed state for Lolo.

Maps the game grid to a simple categorical representation:
  0 = walkable  (empty, bridge, desert)
  1 = blocked   (rock, tree, water, lava)
  2 = heart     (collectible)
  3 = enemy     (danger zone)
  4 = exit/chest (goal)
  5 = pushable  (emerald, egg)

Produces an 84-dim float vector that works identically whether
reading from the simulator or from NES RAM bytes.

The compressed state is quantized so it can be used as a Q-table key
(tuple of rounded floats) for cross-puzzle generalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Tile category mapping (sim tile type → compressed)
_TILE_CATEGORY = {
    0: 0,   # EMPTY → walkable
    1: 1,   # ROCK → blocked
    2: 1,   # TREE → blocked
    3: 2,   # HEART → collectible
    4: 5,   # EMERALD → pushable
    5: 4,   # CHEST → goal
    6: 4,   # EXIT → goal
    7: 1,   # WATER → blocked
    8: 1,   # LAVA → blocked
    9: 0,   # DESERT → walkable
    10: 0,  # BRIDGE → walkable
    11: 1,  # FLOWER → blocked
    12: 0,  # ARROW_UP → walkable
    13: 0,  # ARROW_DOWN → walkable
    14: 0,  # ARROW_LEFT → walkable
    15: 0,  # ARROW_RIGHT → walkable
    16: 0,  # PLAYER → walkable (player tile itself)
    17: 5,  # EGG → pushable
}


class LoloCompressedState:
    """
    Compressed state encoder for Lolo puzzles.

    Produces an 84-dim vector:
      [0:49]   7x7 local grid (categorical: 0-5, normalized to 0-1)
      [49:51]  player position (row/13, col/11)
      [51:54]  hearts (collected/total, chest_open, has_jewel)
      [54:55]  magic shots (shots/10)
      [55:71]  nearest 4 enemies (dx, dy, facing/4, type/7) × 4
      [71:75]  directional danger (LOS exposure per direction)
      [75:79]  walkable path length (4 directions)
      [79:83]  nearest heart/exit direction (dx, dy for each)
      [83:84]  progress (step_count/1000)

    Quantized to 1 decimal for Q-table hashing.
    """

    GRID_H = 13
    GRID_W = 11
    VIEW_RADIUS = 3  # 7x7 local view
    STATE_DIM = 84

    def encode_from_sim(self, sim) -> np.ndarray:
        """Encode from a LoloSimulator instance."""
        return self._encode(
            grid=sim.grid,
            player_row=sim.player_row,
            player_col=sim.player_col,
            hearts_collected=sim.hearts_collected,
            hearts_total=sim.hearts_total,
            chest_open=sim.chest_open,
            has_jewel=sim.has_jewel,
            magic_shots=sim.magic_shots,
            step_count=sim.step_count,
            enemies=[(e.row, e.col, e.facing, e.etype.value,
                       e.alive, e.is_egg, e.is_dangerous)
                      for e in sim.enemies],
        )

    def encode_from_ram(self, ram_state: Dict[str, Any]) -> np.ndarray:
        """
        Encode from NES RAM bytes (same output as encode_from_sim).

        ram_state expected keys:
          grid: 13x11 array of tile values
          player_row, player_col: int
          hearts_collected, hearts_total: int
          chest_open, has_jewel: bool
          magic_shots, step_count: int
          enemies: list of (row, col, facing, type, alive, is_egg, dangerous)
        """
        return self._encode(**ram_state)

    def _encode(
        self,
        grid: np.ndarray,
        player_row: int,
        player_col: int,
        hearts_collected: int,
        hearts_total: int,
        chest_open: bool,
        has_jewel: bool,
        magic_shots: int,
        step_count: int,
        enemies: list,
    ) -> np.ndarray:
        """Core encoder: grid + state → 84-dim float vector."""
        vec = np.zeros(self.STATE_DIM, dtype=np.float32)
        idx = 0

        # ── 1. Local 7×7 grid (49 values) ──
        for dr in range(-self.VIEW_RADIUS, self.VIEW_RADIUS + 1):
            for dc in range(-self.VIEW_RADIUS, self.VIEW_RADIUS + 1):
                r, c = player_row + dr, player_col + dc
                if 0 <= r < self.GRID_H and 0 <= c < self.GRID_W:
                    tile = int(grid[r, c])
                    # Check if enemy is here
                    enemy_here = any(
                        er == r and ec == c and alive and not egg
                        for er, ec, _, _, alive, egg, _ in enemies
                    )
                    if enemy_here:
                        vec[idx] = 3.0 / 5.0  # enemy
                    else:
                        vec[idx] = _TILE_CATEGORY.get(tile, 1) / 5.0
                else:
                    vec[idx] = 1.0 / 5.0  # Out of bounds = blocked
                idx += 1

        # ── 2. Player position (2) ──
        vec[idx] = player_row / self.GRID_H
        vec[idx + 1] = player_col / self.GRID_W
        idx += 2

        # ── 3. Hearts/progress (3) ──
        vec[idx] = hearts_collected / max(1, hearts_total)
        vec[idx + 1] = float(chest_open)
        vec[idx + 2] = float(has_jewel)
        idx += 3

        # ── 4. Magic shots (1) ──
        vec[idx] = min(magic_shots / 10.0, 1.0)
        idx += 1

        # ── 5. Nearest 4 enemies (16) ──
        alive_enemies = [
            (er, ec, ef, et)
            for er, ec, ef, et, alive, egg, dangerous in enemies
            if alive and not egg
        ]
        # Sort by distance to player
        alive_enemies.sort(
            key=lambda e: abs(e[0] - player_row) + abs(e[1] - player_col)
        )

        for i in range(4):
            if i < len(alive_enemies):
                er, ec, ef, et = alive_enemies[i]
                vec[idx] = (er - player_row) / self.GRID_H  # relative dy
                vec[idx + 1] = (ec - player_col) / self.GRID_W  # relative dx
                vec[idx + 2] = ef / 4.0  # facing direction
                vec[idx + 3] = et / 7.0  # enemy type
            # else: zeros (no enemy)
            idx += 4

        # ── 6. Directional danger (4) — LOS exposure ──
        # For each direction, count how many enemies have LOS
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP DOWN LEFT RIGHT
        for dy, dx in dirs:
            danger = 0.0
            r, c = player_row + dy, player_col + dx
            steps = 0
            while 0 <= r < self.GRID_H and 0 <= c < self.GRID_W and steps < 10:
                tile = int(grid[r, c])
                if _TILE_CATEGORY.get(tile, 1) == 1:  # blocked
                    break
                # Check for enemy on this line
                for er, ec, _, _, alive, egg, dangerous in enemies:
                    if er == r and ec == c and alive and not egg and dangerous:
                        danger += 1.0
                r += dy
                c += dx
                steps += 1
            vec[idx] = min(danger / 3.0, 1.0)
            idx += 1

        # ── 7. Walkable path length (4) ──
        for dy, dx in dirs:
            steps = 0
            r, c = player_row + dy, player_col + dx
            while 0 <= r < self.GRID_H and 0 <= c < self.GRID_W and steps < 10:
                tile = int(grid[r, c])
                if _TILE_CATEGORY.get(tile, 1) == 1:  # blocked
                    break
                steps += 1
                r += dy
                c += dx
            vec[idx] = steps / 10.0
            idx += 1

        # ── 8. Nearest heart/exit direction (4) ──
        # Nearest heart
        best_heart_dr, best_heart_dc = 0.0, 0.0
        best_heart_dist = 999
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if int(grid[r, c]) == 3:  # HEART
                    dist = abs(r - player_row) + abs(c - player_col)
                    if dist < best_heart_dist:
                        best_heart_dist = dist
                        best_heart_dr = (r - player_row) / self.GRID_H
                        best_heart_dc = (c - player_col) / self.GRID_W
        vec[idx] = best_heart_dr
        vec[idx + 1] = best_heart_dc

        # Nearest exit/chest
        best_goal_dr, best_goal_dc = 0.0, 0.0
        best_goal_dist = 999
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                tile = int(grid[r, c])
                if tile in (5, 6):  # CHEST or EXIT
                    dist = abs(r - player_row) + abs(c - player_col)
                    if dist < best_goal_dist:
                        best_goal_dist = dist
                        best_goal_dr = (r - player_row) / self.GRID_H
                        best_goal_dc = (c - player_col) / self.GRID_W
        vec[idx + 2] = best_goal_dr
        vec[idx + 3] = best_goal_dc
        idx += 4

        # ── 9. Progress (1) ──
        vec[idx] = min(step_count / 1000.0, 1.0)

        return vec

    def quantize(self, vec: np.ndarray, precision: int = 1) -> tuple:
        """Quantize to hashable tuple for Q-table key."""
        return tuple(np.round(vec, precision).astype(np.float32))

    def encode_key(self, sim) -> tuple:
        """One-shot: encode + quantize for Q-table lookup."""
        return self.quantize(self.encode_from_sim(sim))
