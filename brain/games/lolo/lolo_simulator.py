"""
lolo_simulator.py — Pure Python Adventures of Lolo grid engine.

Simulates the core puzzle mechanics of Adventures of Lolo without any
NES emulation. Runs 100,000x faster than real NES, enabling massive
training through procedural puzzle generation.

Grid: 11 columns × 13 rows
Actions: NOOP(0), UP(1), DOWN(2), LEFT(3), RIGHT(4), SHOOT(5)
Win: Collect all hearts → open chest → grab jewel → reach exit

Usage:
    from brain.games.lolo.lolo_simulator import LoloSimulator, Tile

    grid = np.zeros((13, 11), dtype=np.uint8)
    grid[6, 5] = Tile.PLAYER
    grid[3, 8] = Tile.HEART
    grid[1, 5] = Tile.CHEST
    grid[0, 5] = Tile.EXIT

    sim = LoloSimulator(grid)
    obs, reward, done, info = sim.step(Action.UP)
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Tile types ─────────────────────────────────────────────────────────

class Tile(IntEnum):
    EMPTY       = 0
    ROCK        = 1   # Impassable by all. Breakable with Hammer.
    TREE        = 2   # Blocks movement, some attacks pass through
    HEART       = 3   # Collectible. Some grant magic shots.
    EMERALD     = 4   # Pushable block
    CHEST       = 5   # Opens when all hearts collected
    EXIT        = 6   # Level exit (after chest collected)
    WATER       = 7   # Deadly unless bridged or egg-raft
    LAVA        = 8   # Deadly, eggs sink, only bridge works
    DESERT      = 9   # Passable but slow
    BRIDGE      = 10  # Passable crossing over water/lava
    ARROW_UP    = 11  # One-way pass
    ARROW_DOWN  = 12
    ARROW_LEFT  = 13
    ARROW_RIGHT = 14
    FLOWER      = 15  # Safe from some enemies
    PLAYER      = 16  # Player starting position (converted to EMPTY)
    EGG         = 17  # Transformed enemy, pushable


class Action(IntEnum):
    NOOP  = 0
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4
    SHOOT = 5


# Direction vectors: (dy, dx) for UP, DOWN, LEFT, RIGHT
DIRS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
}


# ── Enemy types ────────────────────────────────────────────────────────

class EnemyType(IntEnum):
    SNAKEY     = 0   # Stationary, harmless, eggable
    LEEPER     = 1   # Sleeps when touched, becomes obstacle
    ROCKY      = 2   # Charges at player in LOS
    ALMA       = 3   # Chases player
    MEDUSA     = 4   # Stationary, kills in LOS, NOT eggable
    DON_MEDUSA = 5   # Moving, kills in LOS, NOT eggable
    GOL        = 6   # Activates after all hearts, shoots in LOS
    SKULL      = 7   # Activates after all hearts, chases

# Which enemies can be turned to eggs
EGGABLE = {EnemyType.SNAKEY, EnemyType.LEEPER, EnemyType.ROCKY,
           EnemyType.ALMA, EnemyType.GOL, EnemyType.SKULL}

# Which enemies kill on line-of-sight (not contact)
LOS_KILLERS = {EnemyType.MEDUSA, EnemyType.DON_MEDUSA, EnemyType.GOL}

# Which enemies activate only after all hearts collected
LATE_ACTIVATE = {EnemyType.GOL, EnemyType.SKULL}


@dataclass
class Enemy:
    """A game enemy with type, position, and state."""
    etype: EnemyType
    row: int
    col: int
    alive: bool = True
    is_egg: bool = False
    egg_timer: int = 0       # Egg reverts after this many steps
    asleep: bool = False     # Leeper: permanently asleep
    active: bool = True      # Gol/Skull: inactive until all hearts collected
    facing: int = 2          # Direction enemy faces (Action enum: 1=UP,2=DOWN,3=LEFT,4=RIGHT)

    # Patrol (Don Medusa): oscillates between two positions
    patrol_dir: int = 0      # 0=horizontal, 1=vertical
    patrol_range: int = 2    # How far to patrol
    patrol_step: int = 0     # Current position in patrol
    patrol_forward: bool = True

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)

    @property
    def blocks_movement(self) -> bool:
        """Does this enemy block player movement?"""
        return self.alive and not self.is_egg and not self.asleep

    @property
    def is_dangerous(self) -> bool:
        """Can this enemy kill the player?"""
        if not self.alive or self.is_egg or self.asleep:
            return False
        if not self.active:
            return False
        return True


# ── Passability ────────────────────────────────────────────────────────

SOLID_TILES = {Tile.ROCK, Tile.TREE, Tile.CHEST, Tile.WATER, Tile.LAVA}
DEADLY_TILES = {Tile.WATER, Tile.LAVA}


def _is_passable(tile: int) -> bool:
    """Can the player walk on this tile?"""
    return tile not in SOLID_TILES


def _blocks_los(tile: int) -> bool:
    """Does this tile block line-of-sight for Medusa/Gol?"""
    # Rocks and emeralds block LOS. Trees do NOT block Medusa.
    return tile in {Tile.ROCK, Tile.EMERALD, Tile.EGG}


# ── Main Simulator ────────────────────────────────────────────────────

class LoloSimulator:
    """
    Pure Python Adventures of Lolo game engine.

    Operates on an 11×13 grid. No pixels — just tile logic.
    Supports save/load for rehearsal, solvability checking via BFS.
    """

    GRID_H = 13  # rows
    GRID_W = 11  # columns

    # Rewards
    R_HEART = 1.0
    R_WIN = 10.0
    R_DEATH = -5.0
    R_STEP = -0.01  # Small negative to encourage efficiency

    # Egg duration (steps before it reverts)
    EGG_DURATION = 30

    def __init__(
        self,
        grid: np.ndarray,
        enemies: Optional[List[Enemy]] = None,
        magic_shot_hearts: Optional[Set[Tuple[int, int]]] = None,
    ):
        """
        Args:
            grid: (GRID_H, GRID_W) uint8 array of Tile values
            enemies: List of Enemy objects
            magic_shot_hearts: Set of (row, col) positions of hearts that grant shots
        """
        assert grid.shape == (self.GRID_H, self.GRID_W), \
            f"Grid must be ({self.GRID_H}, {self.GRID_W}), got {grid.shape}"

        self.grid = grid.astype(np.uint8).copy()
        self.enemies = deepcopy(enemies) if enemies else []
        self._magic_shot_hearts = magic_shot_hearts or set()

        # Find and remove player marker from grid
        player_pos = np.argwhere(self.grid == Tile.PLAYER)
        if len(player_pos) > 0:
            self.player_row = int(player_pos[0][0])
            self.player_col = int(player_pos[0][1])
            self.grid[self.player_row, self.player_col] = Tile.EMPTY
        else:
            self.player_row = self.GRID_H // 2
            self.player_col = self.GRID_W // 2

        # Game state
        self.hearts_total = int(np.sum(self.grid == Tile.HEART))
        self.hearts_collected = 0
        self.magic_shots = 0
        self.chest_open = False
        self.has_jewel = False
        self.alive = True
        self.won = False
        self.step_count = 0
        self.facing = Action.UP  # Direction player is facing (for shooting)

        # Mark late-activate enemies as inactive
        for e in self.enemies:
            if e.etype in LATE_ACTIVATE:
                e.active = False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Returns:
            obs: flattened grid state
            reward: scalar reward
            done: episode over (win or death)
            info: diagnostic dict
        """
        if not self.alive or self.won:
            return self.get_obs(), 0.0, True, {"reason": "already_done"}

        self.step_count += 1
        reward = self.R_STEP
        info: Dict[str, Any] = {}

        action = Action(action)
        self.facing = action if action in DIRS else self.facing

        # ── 1. Handle shooting ────────────────────────────────────────
        if action == Action.SHOOT:
            if self.magic_shots > 0:
                hit = self._shoot(self.facing)
                self.magic_shots -= 1
                info["shot"] = True
                info["shot_hit"] = hit
            else:
                info["shot"] = False

        # ── 2. Handle movement ────────────────────────────────────────
        elif action in DIRS:
            dy, dx = DIRS[action]
            new_r = self.player_row + dy
            new_c = self.player_col + dx

            if self._in_bounds(new_r, new_c):
                tile = self.grid[new_r, new_c]

                # Check one-way arrows
                if tile == Tile.ARROW_UP and action != Action.UP:
                    pass  # Blocked
                elif tile == Tile.ARROW_DOWN and action != Action.DOWN:
                    pass
                elif tile == Tile.ARROW_LEFT and action != Action.LEFT:
                    pass
                elif tile == Tile.ARROW_RIGHT and action != Action.RIGHT:
                    pass

                # Push emerald/egg
                elif tile in (Tile.EMERALD, Tile.EGG):
                    push_r = new_r + dy
                    push_c = new_c + dx
                    if (self._in_bounds(push_r, push_c)
                            and _is_passable(self.grid[push_r, push_c])
                            and not self._enemy_at(push_r, push_c)):
                        # Check if pushing egg into water
                        if tile == Tile.EGG and self.grid[push_r, push_c] == Tile.WATER:
                            self.grid[push_r, push_c] = Tile.BRIDGE  # Egg becomes raft
                            self.grid[new_r, new_c] = Tile.EMPTY
                        else:
                            self.grid[push_r, push_c] = tile
                            self.grid[new_r, new_c] = Tile.EMPTY
                        self.player_row = new_r
                        self.player_col = new_c

                # Collect heart
                elif tile == Tile.HEART:
                    self.grid[new_r, new_c] = Tile.EMPTY
                    self.hearts_collected += 1
                    reward += self.R_HEART
                    info["collected_heart"] = True

                    # Check if this heart grants magic shots
                    if (new_r, new_c) in self._magic_shot_hearts:
                        self.magic_shots += 2
                        info["got_magic_shots"] = True

                    # All hearts collected → open chest, activate late enemies
                    if self.hearts_collected >= self.hearts_total:
                        self.chest_open = True
                        info["chest_opened"] = True
                        for e in self.enemies:
                            if e.etype in LATE_ACTIVATE and e.alive:
                                e.active = True

                    self.player_row = new_r
                    self.player_col = new_c

                # Enter open chest
                elif tile == Tile.CHEST and self.chest_open:
                    self.has_jewel = True
                    self.grid[new_r, new_c] = Tile.EMPTY
                    reward += self.R_HEART
                    info["got_jewel"] = True
                    self.player_row = new_r
                    self.player_col = new_c

                # Reach exit with jewel
                elif tile == Tile.EXIT and self.has_jewel:
                    self.won = True
                    reward += self.R_WIN
                    info["won"] = True
                    self.player_row = new_r
                    self.player_col = new_c

                # Normal passable tile
                elif _is_passable(tile):
                    self.player_row = new_r
                    self.player_col = new_c

                # Deadly tile
                elif tile in DEADLY_TILES:
                    self.alive = False
                    reward += self.R_DEATH
                    info["death"] = "drowned"

        # ── 3. Update enemies ─────────────────────────────────────────
        self._update_enemies()

        # ── 4. Check enemy interactions ───────────────────────────────
        for e in self.enemies:
            if not e.is_dangerous:
                continue

            # Contact kill (Alma, Skull, Rocky charge)
            if e.etype in (EnemyType.ALMA, EnemyType.SKULL):
                if e.row == self.player_row and e.col == self.player_col:
                    self.alive = False
                    reward += self.R_DEATH
                    info["death"] = f"killed_by_{e.etype.name}"
                    break

            # Leeper: touching puts it to sleep
            if e.etype == EnemyType.LEEPER and not e.asleep:
                if e.row == self.player_row and e.col == self.player_col:
                    e.asleep = True
                    info["leeper_asleep"] = True

            # LOS killers (Medusa/Don Medusa: facing direction only; Gol: all 4)
            if e.etype in LOS_KILLERS and e.active:
                facing_constraint = e.facing if e.etype in (EnemyType.MEDUSA, EnemyType.DON_MEDUSA) else None
                if self._has_los(e.row, e.col, self.player_row, self.player_col, facing_constraint):
                    self.alive = False
                    reward += self.R_DEATH
                    info["death"] = f"los_killed_by_{e.etype.name}"
                    break

        # ── 5. Update egg timers ──────────────────────────────────────
        for e in self.enemies:
            if e.is_egg:
                e.egg_timer -= 1
                if e.egg_timer <= 0:
                    e.is_egg = False  # Revert to normal enemy

        done = not self.alive or self.won
        return self.get_obs(), reward, done, info

    # ── Shooting ──────────────────────────────────────────────────────

    def _shoot(self, direction: Action) -> bool:
        """Fire a magic shot in direction. Returns True if hit an enemy."""
        if direction not in DIRS:
            return False
        dy, dx = DIRS[direction]
        r, c = self.player_row + dy, self.player_col + dx

        while self._in_bounds(r, c):
            # Check for enemy
            for e in self.enemies:
                if e.row == r and e.col == c and e.alive and not e.is_egg:
                    if e.etype in EGGABLE:
                        e.is_egg = True
                        e.egg_timer = self.EGG_DURATION
                        # Place egg tile on grid for LOSblocking
                        self.grid[r, c] = Tile.EGG
                        return True
                    else:
                        return False  # Hit uneggable enemy (Medusa)
                # Hit an egg → remove it temporary
                if e.row == r and e.col == c and e.is_egg:
                    e.alive = False
                    self.grid[r, c] = Tile.EMPTY
                    return True

            # Check for blocking tile
            tile = self.grid[r, c]
            if tile in {Tile.ROCK, Tile.TREE, Tile.EMERALD, Tile.EGG}:
                return False  # Shot blocked

            r += dy
            c += dx

        return False  # Shot went off screen

    # ── Enemy AI ──────────────────────────────────────────────────────

    def _update_enemies(self) -> None:
        """Update all enemy positions."""
        for e in self.enemies:
            if not e.alive or e.is_egg or e.asleep or not e.active:
                continue

            if e.etype == EnemyType.SNAKEY:
                pass  # Stationary

            elif e.etype == EnemyType.ROCKY:
                self._ai_rocky(e)

            elif e.etype in (EnemyType.ALMA, EnemyType.SKULL):
                self._ai_chase(e)

            elif e.etype == EnemyType.DON_MEDUSA:
                self._ai_patrol(e)

    def _ai_chase(self, e: Enemy) -> None:
        """Chase player (Alma, Skull). Simple greedy pursuit."""
        dr = self.player_row - e.row
        dc = self.player_col - e.col

        # Prefer axis with larger distance
        moves = []
        if abs(dr) >= abs(dc):
            if dr > 0: moves.append((1, 0))
            elif dr < 0: moves.append((-1, 0))
            if dc > 0: moves.append((0, 1))
            elif dc < 0: moves.append((0, -1))
        else:
            if dc > 0: moves.append((0, 1))
            elif dc < 0: moves.append((0, -1))
            if dr > 0: moves.append((1, 0))
            elif dr < 0: moves.append((-1, 0))

        for dy, dx in moves:
            nr, nc = e.row + dy, e.col + dx
            if (self._in_bounds(nr, nc)
                    and _is_passable(self.grid[nr, nc])
                    and not self._enemy_at(nr, nc)):
                e.row, e.col = nr, nc
                break

    def _ai_rocky(self, e: Enemy) -> None:
        """Rocky charges if player is in line-of-sight on same row/col."""
        if e.row == self.player_row:
            # Same row — charge horizontally
            dc = 1 if self.player_col > e.col else -1
            nr, nc = e.row, e.col + dc
            if (self._in_bounds(nr, nc)
                    and _is_passable(self.grid[nr, nc])
                    and not self._enemy_at(nr, nc)):
                e.col = nc
        elif e.col == self.player_col:
            # Same col — charge vertically
            dr = 1 if self.player_row > e.row else -1
            nr, nc = e.row + dr, e.col
            if (self._in_bounds(nr, nc)
                    and _is_passable(self.grid[nr, nc])
                    and not self._enemy_at(nr, nc)):
                e.row = nr

    def _ai_patrol(self, e: Enemy) -> None:
        """Don Medusa patrols back and forth."""
        if e.patrol_dir == 0:  # Horizontal
            dc = 1 if e.patrol_forward else -1
            nr, nc = e.row, e.col + dc
        else:  # Vertical
            dc = 0
            dr = 1 if e.patrol_forward else -1
            nr, nc = e.row + dr, e.col

        if (self._in_bounds(nr, nc)
                and _is_passable(self.grid[nr, nc])
                and not self._enemy_at(nr, nc)):
            e.row, e.col = nr, nc
            e.patrol_step += 1
            if e.patrol_step >= e.patrol_range:
                e.patrol_forward = not e.patrol_forward
                e.patrol_step = 0
        else:
            e.patrol_forward = not e.patrol_forward
            e.patrol_step = 0

    # ── Line of Sight ─────────────────────────────────────────────────

    def _has_los(
        self, er: int, ec: int, pr: int, pc: int,
        facing: Optional[int] = None,
    ) -> bool:
        """
        Does enemy at (er,ec) have clear LOS to player at (pr,pc)?

        If facing is set (Medusa/Don Medusa), only checks the facing direction.
        facing uses Action enum: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
        """
        # Must be on same row or column
        if er != pr and ec != pc:
            return False

        # Check facing constraint (Medusa only fires in one direction)
        if facing is not None:
            if er == pr:  # Same row → horizontal LOS
                if pc > ec and facing != Action.RIGHT:
                    return False
                if pc < ec and facing != Action.LEFT:
                    return False
            else:  # Same col → vertical LOS
                if pr > er and facing != Action.DOWN:
                    return False
                if pr < er and facing != Action.UP:
                    return False

        if er == pr:
            # Same row — check horizontal
            c_min, c_max = min(ec, pc), max(ec, pc)
            for c in range(c_min + 1, c_max):
                if _blocks_los(self.grid[er, c]) or self._enemy_at(er, c):
                    return False
            return True
        else:
            # Same col — check vertical
            r_min, r_max = min(er, pr), max(er, pr)
            for r in range(r_min + 1, r_max):
                if _blocks_los(self.grid[r, ec]) or self._enemy_at(r, ec):
                    return False
            return True

    # ── Utilities ─────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.GRID_H and 0 <= c < self.GRID_W

    def _enemy_at(self, r: int, c: int) -> bool:
        """Is there a blocking enemy at this position?"""
        for e in self.enemies:
            if e.row == r and e.col == c and e.blocks_movement:
                return True
        return False

    # ── Observation ───────────────────────────────────────────────────

    def get_obs(self) -> np.ndarray:
        """
        Return game state as flat numpy array.

        Layout: grid flat (143) + player_pos (2) + hearts (3) + shots (1)
                + enemy_states (8 * n_enemies) + flags (4)
        """
        obs_parts = [
            self.grid.flatten().astype(np.float32) / 17.0,  # Normalized tiles
            np.array([self.player_row / self.GRID_H,
                       self.player_col / self.GRID_W], dtype=np.float32),
            np.array([self.hearts_collected / max(1, self.hearts_total),
                       float(self.chest_open),
                       float(self.has_jewel)], dtype=np.float32),
            np.array([self.magic_shots / 10.0], dtype=np.float32),
        ]

        # Enemy states (up to 8 enemies, pad if fewer)
        for i in range(8):
            if i < len(self.enemies):
                e = self.enemies[i]
                obs_parts.append(np.array([
                    e.etype / 7.0,
                    e.row / self.GRID_H,
                    e.col / self.GRID_W,
                    float(e.alive),
                    float(e.is_egg),
                    float(e.active),
                    float(e.asleep),
                    float(e.is_dangerous),
                ], dtype=np.float32))
            else:
                obs_parts.append(np.zeros(8, dtype=np.float32))

        obs_parts.append(np.array([
            float(self.alive),
            float(self.won),
            self.step_count / 1000.0,
            float(self.facing) / 5.0,
        ], dtype=np.float32))

        return np.concatenate(obs_parts)

    @property
    def obs_size(self) -> int:
        """Expected observation vector size."""
        return 143 + 2 + 3 + 1 + 8 * 8 + 4  # = 217

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self) -> Dict[str, Any]:
        """Serialize full state for rehearsal save/load."""
        return {
            "grid": self.grid.copy(),
            "player_row": self.player_row,
            "player_col": self.player_col,
            "hearts_collected": self.hearts_collected,
            "hearts_total": self.hearts_total,
            "magic_shots": self.magic_shots,
            "chest_open": self.chest_open,
            "has_jewel": self.has_jewel,
            "alive": self.alive,
            "won": self.won,
            "step_count": self.step_count,
            "facing": int(self.facing),
            "magic_shot_hearts": [list(pos) for pos in self._magic_shot_hearts],
            "enemies": [
                {
                    "etype": int(e.etype), "row": e.row, "col": e.col,
                    "alive": e.alive, "is_egg": e.is_egg,
                    "egg_timer": e.egg_timer, "asleep": e.asleep,
                    "active": e.active, "facing": e.facing,
                    "patrol_dir": e.patrol_dir,
                    "patrol_range": e.patrol_range,
                    "patrol_step": e.patrol_step,
                    "patrol_forward": e.patrol_forward,
                }
                for e in self.enemies
            ],
        }

    def load(self, state: Dict[str, Any]) -> None:
        """Restore full state from save."""
        self.grid = state["grid"].copy()
        self.player_row = state["player_row"]
        self.player_col = state["player_col"]
        self.hearts_collected = state["hearts_collected"]
        self.hearts_total = state["hearts_total"]
        self.magic_shots = state["magic_shots"]
        self.chest_open = state["chest_open"]
        self.has_jewel = state["has_jewel"]
        self.alive = state["alive"]
        self.won = state["won"]
        self.step_count = state["step_count"]
        self.facing = Action(state["facing"])
        self._magic_shot_hearts = set(
            tuple(pos) for pos in state.get("magic_shot_hearts", [])
        )
        self.enemies = []
        for ed in state["enemies"]:
            e = Enemy(
                etype=EnemyType(ed["etype"]),
                row=ed["row"], col=ed["col"],
                alive=ed["alive"], is_egg=ed["is_egg"],
                egg_timer=ed["egg_timer"], asleep=ed["asleep"],
                active=ed["active"], facing=ed.get("facing", 2),
                patrol_dir=ed["patrol_dir"],
                patrol_range=ed["patrol_range"],
                patrol_step=ed["patrol_step"],
                patrol_forward=ed["patrol_forward"],
            )
            self.enemies.append(e)

    # ── Solvability ───────────────────────────────────────────────────

    def is_solvable(self) -> bool:
        """
        Backward goal validation:
          1. Exit must be reachable from player
          2. Chest must be reachable from player
          3. All hearts must be reachable from player
          4. Every LOS hazard (Medusa/DonMedusa) must either:
             (a) NOT cover the path from player to exit/chest/any heart, OR
             (b) Have a blockable tile (Emerald/existing Rock) that could
                 shield the required path
        """
        reachable = self._bfs_reachable(self.player_row, self.player_col)

        # ── Check 1: Exit reachable ───────────────────────────────────
        exit_positions = list(zip(*np.where(self.grid == Tile.EXIT)))
        if not exit_positions:
            return False
        for er, ec in exit_positions:
            if (er, ec) not in reachable and not self._adjacent_reachable(er, ec, reachable):
                return False

        # ── Check 2: Chest reachable ──────────────────────────────────
        chest_positions = list(zip(*np.where(self.grid == Tile.CHEST)))
        if not chest_positions:
            return False
        for cr, cc in chest_positions:
            if (cr, cc) not in reachable and not self._adjacent_reachable(cr, cc, reachable):
                return False

        # ── Check 3: All hearts reachable ─────────────────────────────
        heart_positions = list(zip(*np.where(self.grid == Tile.HEART)))
        for hr, hc in heart_positions:
            if (hr, hc) not in reachable:
                return False

        # ── Check 4: LOS hazards must be blockable or avoidable ───────
        critical_cells = set()
        critical_cells.add((self.player_row, self.player_col))
        for hr, hc in heart_positions:
            critical_cells.add((hr, hc))
        for cr, cc in chest_positions:
            critical_cells.add((cr, cc))
        for er, ec in exit_positions:
            critical_cells.add((er, ec))

        for e in self.enemies:
            if e.etype not in LOS_KILLERS:
                continue
            if not e.alive:
                continue

            # Check each LOS direction from this enemy
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Walk the LOS lane. A shield (Rock/Emerald) only counts
                # if it appears BEFORE any critical cell in the lane.
                r, c = e.row + dr, e.col + dc
                shielded = False      # Found a shield before any critical cell
                exposed = False       # Found a critical cell with no prior shield

                while self._in_bounds(r, c):
                    tile = self.grid[r, c]

                    # Rock/Emerald blocks LOS — if we haven't hit a critical
                    # cell yet, everything beyond is shielded
                    if tile in (Tile.ROCK, Tile.EMERALD):
                        if not exposed:
                            shielded = True
                        break  # Nothing beyond matters for this lane

                    # Tree does NOT block Medusa LOS (real Lolo rule)

                    # Critical cell in the lane with no shield before it
                    if (r, c) in critical_cells:
                        exposed = True

                    r += dr
                    c += dc

                # If a critical cell is exposed (no shield before it)
                if exposed and not shielded:
                    # Last chance: can player push an Emerald into this lane?
                    if not self._can_shield_lane(e.row, e.col, dr, dc, reachable):
                        return False

        return True

    def _adjacent_reachable(
        self, r: int, c: int, reachable: Set[Tuple[int, int]],
    ) -> bool:
        """Is any cell adjacent to (r,c) in the reachable set?"""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in reachable:
                return True
        return False

    def _can_shield_lane(
        self, er: int, ec: int, dr: int, dc: int,
        reachable: Set[Tuple[int, int]],
    ) -> bool:
        """
        Can the player push an Emerald into this LOS lane to block it?

        Checks if there's a reachable Emerald adjacent to any cell in the lane
        such that pushing it into the lane is geometrically possible.
        """
        # Scan the lane cells
        r, c = er + dr, ec + dc
        while self._in_bounds(r, c):
            tile = self.grid[r, c]
            if tile == Tile.ROCK:
                break  # Lane is already blocked

            # Can an emerald be pushed here from a perpendicular direction?
            for push_dr, push_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # The emerald would be at (r - push_dr, c - push_dc)
                # and pushed from (r - 2*push_dr, c - 2*push_dc)
                em_r, em_c = r - push_dr, c - push_dc
                push_from_r, push_from_c = r - 2 * push_dr, c - 2 * push_dc

                if (self._in_bounds(em_r, em_c)
                        and self.grid[em_r, em_c] == Tile.EMERALD
                        and self._in_bounds(push_from_r, push_from_c)
                        and (push_from_r, push_from_c) in reachable):
                    return True

            r += dr
            c += dc

        return False

    def is_dead_end(self) -> bool:
        """
        Runtime dead-end detection for during play.

        Called periodically to detect if the current state is unwinnable.
        If True, the episode should end early (agent "gives up").
        """
        # If already won or dead, not a dead-end per se
        if self.won or not self.alive:
            return False

        reachable = self._bfs_reachable(self.player_row, self.player_col)

        # Can player reach remaining hearts?
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if self.grid[r, c] == Tile.HEART and (r, c) not in reachable:
                    return True  # Unreachable heart = dead end

        # If all hearts collected, can player reach chest?
        if self.hearts_collected >= self.hearts_total and not self.has_jewel:
            chest_found = False
            for r in range(self.GRID_H):
                for c in range(self.GRID_W):
                    if self.grid[r, c] == Tile.CHEST:
                        if self._adjacent_reachable(r, c, reachable):
                            chest_found = True
                        break
            if not chest_found:
                return True

        # If has jewel, can player reach exit?
        if self.has_jewel:
            exit_found = False
            for r in range(self.GRID_H):
                for c in range(self.GRID_W):
                    if self.grid[r, c] == Tile.EXIT:
                        if self._adjacent_reachable(r, c, reachable):
                            exit_found = True
                        break
            if not exit_found:
                return True

        return False

    def _bfs_reachable(self, start_r: int, start_c: int) -> Set[Tuple[int, int]]:
        """BFS flood fill from start position. Returns set of reachable (r,c)."""
        visited: Set[Tuple[int, int]] = set()
        queue: deque = deque()
        queue.append((start_r, start_c))
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self._in_bounds(nr, nc):
                    tile = self.grid[nr, nc]
                    # Can pass through: empty, heart, desert, flower, bridge, arrows
                    if tile not in {Tile.ROCK, Tile.TREE, Tile.WATER,
                                    Tile.LAVA, Tile.EMERALD, Tile.EGG}:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return visited

    # ── Display ───────────────────────────────────────────────────────

    TILE_CHARS = {
        Tile.EMPTY: ".", Tile.ROCK: "#", Tile.TREE: "T",
        Tile.HEART: "H", Tile.EMERALD: "B", Tile.CHEST: "C",
        Tile.EXIT: "E", Tile.WATER: "~", Tile.LAVA: "^",
        Tile.DESERT: ",", Tile.BRIDGE: "=", Tile.FLOWER: "*",
        Tile.ARROW_UP: "u", Tile.ARROW_DOWN: "d",
        Tile.ARROW_LEFT: "l", Tile.ARROW_RIGHT: "r",
        Tile.EGG: "O",
    }

    ENEMY_CHARS = {
        EnemyType.SNAKEY: "s", EnemyType.LEEPER: "l",
        EnemyType.ROCKY: "r", EnemyType.ALMA: "a",
        EnemyType.GOL: "G", EnemyType.SKULL: "S",
    }

    # Medusa/Don Medusa use facing-dependent chars
    _MEDUSA_FACING = {1: "▲", 2: "▼", 3: "◄", 4: "►"}       # UP DOWN LEFT RIGHT
    _DON_MEDUSA_FACING = {1: "△", 2: "▽", 3: "◁", 4: "▷"}

    def _enemy_char(self, e: 'Enemy') -> str:
        """Get display character for an enemy, using facing arrows for Medusa."""
        if e.is_egg:
            return "O"
        if e.asleep:
            return "z"
        if e.etype == EnemyType.MEDUSA:
            return self._MEDUSA_FACING.get(e.facing, "M")
        if e.etype == EnemyType.DON_MEDUSA:
            return self._DON_MEDUSA_FACING.get(e.facing, "D")
        return self.ENEMY_CHARS.get(e.etype, "?")

    def render_ascii(self) -> str:
        """Render grid as ASCII art."""
        lines = []
        for r in range(self.GRID_H):
            row_chars = []
            for c in range(self.GRID_W):
                if r == self.player_row and c == self.player_col:
                    row_chars.append("@")
                else:
                    enemy_here = None
                    for e in self.enemies:
                        if e.row == r and e.col == c and e.alive:
                            enemy_here = e
                            break
                    if enemy_here:
                        row_chars.append(self._enemy_char(enemy_here))
                    else:
                        row_chars.append(
                            self.TILE_CHARS.get(Tile(self.grid[r, c]), "?")
                        )
            lines.append(" ".join(row_chars))

        header = (
            f"Step:{self.step_count} Hearts:{self.hearts_collected}/{self.hearts_total}"
            f" Shots:{self.magic_shots} {'CHEST OPEN' if self.chest_open else ''}"
            f" {'JEWEL' if self.has_jewel else ''}"
        )
        return header + "\n" + "\n".join(lines)
