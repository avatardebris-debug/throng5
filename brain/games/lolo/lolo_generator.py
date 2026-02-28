"""
lolo_generator.py — Procedural puzzle generation for Adventures of Lolo.

Generates solvable Lolo puzzles at controllable complexity tiers.
Each tier introduces new object types, teaching the agent incrementally:

  Tier 1: Empty + Rock + Heart + Chest + Exit (basic navigation)
  Tier 2: + Emerald Framer (pushable blocks)
  Tier 3: + Medusa / Don Medusa (line-of-sight danger)
  Tier 4: + Snakey / Leeper (harmless enemies, manipulation)
  Tier 5: + Rocky / Alma (active threats)
  Tier 6: + Water / Egg-as-raft / Bridge
  Tier 7: + Gol / Skull (endgame pressure)

Usage:
    gen = LoloPuzzleGenerator(seed=42)
    puzzle = gen.generate(tier=1)    # Simple navigation puzzle
    print(puzzle.render_ascii())
    
    batch = gen.generate_batch(100, tier=3)  # 100 tier-3 puzzles
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from brain.games.lolo.lolo_simulator import (
    Action,
    Enemy,
    EnemyType,
    LoloSimulator,
    Tile,
)


# ── Tier configurations ───────────────────────────────────────────────

TIER_CONFIG = {
    1: {
        "name": "basic_navigation",
        "tiles": [Tile.ROCK],
        "enemies": [],
        "hearts": (2, 3),      # (min, max) hearts
        "rocks": (3, 8),       # (min, max) rocks
        "desc": "Navigate, collect hearts, reach exit",
    },
    2: {
        "name": "pushable_blocks",
        "tiles": [Tile.ROCK, Tile.EMERALD],
        "enemies": [],
        "hearts": (2, 4),
        "rocks": (3, 6),
        "emeralds": (1, 3),
        "desc": "Use pushable blocks to create paths",
    },
    3: {
        "name": "line_of_sight",
        "tiles": [Tile.ROCK, Tile.EMERALD, Tile.TREE],
        "enemies": [EnemyType.MEDUSA],
        "hearts": (2, 4),
        "rocks": (3, 6),
        "emeralds": (1, 2),
        "trees": (1, 3),
        "n_enemies": (1, 2),
        "desc": "Avoid Medusa line-of-sight, use blocks as shields",
    },
    4: {
        "name": "enemy_manipulation",
        "tiles": [Tile.ROCK, Tile.EMERALD, Tile.TREE],
        "enemies": [EnemyType.SNAKEY, EnemyType.LEEPER],
        "hearts": (3, 5),
        "rocks": (3, 6),
        "emeralds": (1, 3),
        "trees": (1, 2),
        "n_enemies": (1, 3),
        "magic_shot_chance": 0.5,
        "desc": "Turn enemies to eggs, use as tools",
    },
    5: {
        "name": "active_threats",
        "tiles": [Tile.ROCK, Tile.EMERALD, Tile.TREE],
        "enemies": [EnemyType.ROCKY, EnemyType.ALMA, EnemyType.MEDUSA],
        "hearts": (3, 5),
        "rocks": (4, 8),
        "emeralds": (1, 3),
        "trees": (1, 3),
        "n_enemies": (2, 4),
        "magic_shot_chance": 0.4,
        "desc": "Enemies that move and chase",
    },
    6: {
        "name": "water_mechanics",
        "tiles": [Tile.ROCK, Tile.EMERALD, Tile.TREE, Tile.WATER],
        "enemies": [EnemyType.SNAKEY, EnemyType.MEDUSA],
        "hearts": (3, 5),
        "rocks": (3, 6),
        "emeralds": (1, 2),
        "water": (2, 5),
        "n_enemies": (1, 3),
        "magic_shot_chance": 0.6,
        "desc": "Use eggs to cross water, complex paths",
    },
    7: {
        "name": "endgame_pressure",
        "tiles": [Tile.ROCK, Tile.EMERALD, Tile.TREE, Tile.WATER],
        "enemies": [EnemyType.MEDUSA, EnemyType.GOL, EnemyType.SKULL,
                    EnemyType.SNAKEY, EnemyType.ALMA],
        "hearts": (4, 6),
        "rocks": (4, 8),
        "emeralds": (1, 3),
        "trees": (1, 3),
        "water": (0, 3),
        "n_enemies": (3, 5),
        "magic_shot_chance": 0.5,
        "desc": "Full Lolo: enemies activate after all hearts",
    },
}


class LoloPuzzleGenerator:
    """
    Generate solvable Lolo puzzles at controllable difficulty.

    Uses constraint-satisfaction with random restarts:
    1. Place border walls
    2. Place interior rocks/trees
    3. Place hearts (ensuring reachability)
    4. Place chest and exit
    5. Place enemies in strategic positions
    6. Verify solvability via BFS
    7. If unsolvable, regenerate (max attempts)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.complexity_tier = 1
        self._total_generated = 0
        self._total_attempts = 0

    def generate(
        self,
        tier: Optional[int] = None,
        max_attempts: int = 200,
    ) -> Optional[LoloSimulator]:
        """
        Generate a single solvable puzzle at the given tier.

        Returns None if no solvable puzzle found within max_attempts.
        """
        tier = tier or self.complexity_tier
        config = TIER_CONFIG.get(tier, TIER_CONFIG[1])

        for attempt in range(max_attempts):
            self._total_attempts += 1
            sim = self._try_generate(config, tier)
            if sim is not None and sim.is_solvable():
                self._total_generated += 1
                return sim

        return None

    def generate_batch(
        self,
        n: int,
        tier: Optional[int] = None,
    ) -> List[LoloSimulator]:
        """Generate n solvable puzzles. May return fewer if generation is hard."""
        results = []
        for _ in range(n):
            sim = self.generate(tier)
            if sim is not None:
                results.append(sim)
        return results

    def advance_tier(self) -> int:
        """Advance to next complexity tier. Returns new tier."""
        self.complexity_tier = min(self.complexity_tier + 1, 7)
        return self.complexity_tier

    def _try_generate(self, config: dict, tier: int) -> Optional[LoloSimulator]:
        """Single attempt to generate a puzzle."""
        H, W = LoloSimulator.GRID_H, LoloSimulator.GRID_W
        grid = np.zeros((H, W), dtype=np.uint8)

        # ── 1. Border walls ───────────────────────────────────────────
        grid[0, :] = Tile.ROCK
        grid[H - 1, :] = Tile.ROCK
        grid[:, 0] = Tile.ROCK
        grid[:, W - 1] = Tile.ROCK

        # Interior cells (1..H-2, 1..W-2)
        interior = []
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                interior.append((r, c))
        self.rng.shuffle(interior)

        idx = 0

        def _take(n: int) -> List[Tuple[int, int]]:
            nonlocal idx
            cells = interior[idx:idx + n]
            idx += n
            return cells

        # ── 2. Place rocks ────────────────────────────────────────────
        n_rocks = self.rng.randint(*config.get("rocks", (3, 6)))
        for r, c in _take(n_rocks):
            grid[r, c] = Tile.ROCK

        # ── 3. Place emeralds ─────────────────────────────────────────
        em_range = config.get("emeralds", (0, 1))
        n_emeralds = self.rng.randint(em_range[0], max(em_range[1], em_range[0] + 1))
        for r, c in _take(n_emeralds):
            grid[r, c] = Tile.EMERALD

        # ── 4. Place trees ────────────────────────────────────────────
        tr_range = config.get("trees", (0, 1))
        n_trees = self.rng.randint(tr_range[0], max(tr_range[1], tr_range[0] + 1))
        for r, c in _take(n_trees):
            grid[r, c] = Tile.TREE

        # ── 5. Place water ────────────────────────────────────────────
        wa_range = config.get("water", (0, 1))
        n_water = self.rng.randint(wa_range[0], max(wa_range[1], wa_range[0] + 1))
        for r, c in _take(n_water):
            grid[r, c] = Tile.WATER

        # ── 6. Place player ───────────────────────────────────────────
        player_cell = _take(1)
        if not player_cell:
            return None
        pr, pc = player_cell[0]
        grid[pr, pc] = Tile.PLAYER

        # ── 7. Place hearts ───────────────────────────────────────────
        n_hearts = self.rng.randint(*config.get("hearts", (2, 3)))
        magic_shot_hearts: Set[Tuple[int, int]] = set()
        magic_chance = config.get("magic_shot_chance", 0.0)

        heart_cells = _take(n_hearts)
        for r, c in heart_cells:
            grid[r, c] = Tile.HEART
            if self.rng.random() < magic_chance:
                magic_shot_hearts.add((r, c))

        # ── 8. Place chest ────────────────────────────────────────────
        chest_cell = _take(1)
        if not chest_cell:
            return None
        cr, cc = chest_cell[0]
        grid[cr, cc] = Tile.CHEST

        # ── 9. Place exit ─────────────────────────────────────────────
        # Exit should be near the chest or on a border
        exit_cell = _take(1)
        if not exit_cell:
            return None
        er, ec = exit_cell[0]
        grid[er, ec] = Tile.EXIT

        # ── 10. Place enemies ─────────────────────────────────────────
        enemies: List[Enemy] = []
        enemy_types = config.get("enemies", [])
        if enemy_types:
            n_min, n_max = config.get("n_enemies", (1, 2))
            n_enemies = self.rng.randint(n_min, n_max + 1)
            enemy_cells = _take(n_enemies)

            for r, c in enemy_cells:
                etype = EnemyType(self.rng.choice(enemy_types))
                enemy = Enemy(etype=etype, row=r, col=c)

                # Medusa / Don Medusa: randomize facing direction (1-4)
                if etype in (EnemyType.MEDUSA, EnemyType.DON_MEDUSA):
                    enemy.facing = int(self.rng.randint(1, 5))  # 1=UP,2=DOWN,3=LEFT,4=RIGHT

                # Don Medusa: set patrol parameters
                if etype == EnemyType.DON_MEDUSA:
                    enemy.patrol_dir = int(self.rng.randint(0, 2))
                    enemy.patrol_range = int(self.rng.randint(1, 4))

                enemies.append(enemy)

        # ── Create simulator ──────────────────────────────────────────
        try:
            sim = LoloSimulator(grid, enemies, magic_shot_hearts)
            return sim
        except Exception:
            return None

    def report(self) -> dict:
        return {
            "current_tier": self.complexity_tier,
            "tier_name": TIER_CONFIG.get(
                self.complexity_tier, {}
            ).get("name", "unknown"),
            "total_generated": self._total_generated,
            "total_attempts": self._total_attempts,
            "success_rate": (
                round(self._total_generated / max(1, self._total_attempts), 3)
            ),
        }
