"""
lolo_adapter.py — Bridge between LoloSimulator and WholeBrain.

Converts Lolo grid state into the formats consumed by brain modules:
  1. Flat features (84-dim) for Striatum DQN
  2. ObjectGraph entities for EntityGNN
  3. Fake RAM bytes for RAMSemanticMapper compatibility

Also provides a gymnasium-like interface for integration with
existing training loops.

Usage:
    from brain.games.lolo.lolo_adapter import LoloAdapter

    adapter = LoloAdapter()
    sim = LoloSimulator(grid, enemies)
    features = adapter.grid_to_features(sim)
    graph = adapter.grid_to_object_graph(sim)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.games.lolo.lolo_simulator import (
    Action,
    EnemyType,
    LoloSimulator,
    Tile,
)


# ── Category mapping for GNN ──────────────────────────────────────────

_ENEMY_TO_GNN_CATEGORY = {
    EnemyType.SNAKEY: "enemy",
    EnemyType.LEEPER: "enemy",
    EnemyType.ROCKY: "enemy",
    EnemyType.ALMA: "enemy",
    EnemyType.MEDUSA: "hazard",
    EnemyType.DON_MEDUSA: "hazard",
    EnemyType.GOL: "hazard",
    EnemyType.SKULL: "enemy",
}


class LoloAdapter:
    """
    Bridge LoloSimulator ↔ WholeBrain.

    Provides gymnasium-compatible interface (reset/step/observation)
    and feature extraction for brain modules.
    """

    def __init__(self, feature_dim: int = 84):
        self.feature_dim = feature_dim
        self._sim: Optional[LoloSimulator] = None
        self._step_count = 0

        # Random projection: sim obs (217-dim) → feature_dim
        rng = np.random.RandomState(42)
        self._projection = rng.randn(217, feature_dim).astype(np.float32)
        self._projection /= np.sqrt(217)

        # Running normalization
        self._mean = np.zeros(feature_dim, dtype=np.float32)
        self._var = np.ones(feature_dim, dtype=np.float32)
        self._n_obs = 0

    def set_simulator(self, sim: LoloSimulator) -> None:
        """Attach a simulator instance."""
        self._sim = sim
        self._step_count = 0

    # ── Gymnasium-like interface ──────────────────────────────────────

    def reset(self, sim: LoloSimulator) -> np.ndarray:
        """Reset with a new simulator. Returns initial features."""
        self._sim = sim
        self._step_count = 0
        return self.grid_to_features(sim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the simulator and return (features, reward, done, info)."""
        assert self._sim is not None, "Call reset() first"
        obs, reward, done, info = self._sim.step(action)
        self._step_count += 1
        features = self.grid_to_features(self._sim)
        info["raw_obs"] = obs
        return features, reward, done, info

    # ── Feature extraction ────────────────────────────────────────────

    def grid_to_features(self, sim: LoloSimulator) -> np.ndarray:
        """Convert grid state to 84-dim feature vector via random projection."""
        raw = sim.get_obs()

        # Pad or truncate to projection input size
        if len(raw) > 217:
            raw = raw[:217]
        elif len(raw) < 217:
            raw = np.pad(raw, (0, 217 - len(raw)))

        features = raw @ self._projection

        # Running normalization
        self._n_obs += 1
        alpha = max(0.01, 1.0 / self._n_obs)
        self._mean = (1 - alpha) * self._mean + alpha * features
        self._var = (1 - alpha) * self._var + alpha * (features - self._mean) ** 2
        features = (features - self._mean) / (np.sqrt(self._var) + 1e-8)

        return features.astype(np.float32)

    def grid_to_object_graph(self, sim: LoloSimulator):
        """
        Convert grid state to ObjectGraph for EntityGNN.

        Creates entities for:
        - Player (category="agent")
        - Each heart (category="item")
        - Each enemy (category="enemy" or "hazard")
        - Chest and Exit (category="door")
        - Pushable blocks (category="block")
        """
        try:
            from brain.planning.object_graph import ObjectGraph
        except ImportError:
            return None

        graph = ObjectGraph()

        # Player
        graph.add_entity(
            "player",
            properties={
                "x": sim.player_col * 16,
                "y": sim.player_row * 16,
                "alive": sim.alive,
                "magic_shots": sim.magic_shots,
                "hearts_collected": sim.hearts_collected,
                "has_jewel": sim.has_jewel,
            },
            category="agent",
        )

        # Hearts
        heart_idx = 0
        for r in range(sim.GRID_H):
            for c in range(sim.GRID_W):
                if sim.grid[r, c] == Tile.HEART:
                    graph.add_entity(
                        f"heart_{heart_idx}",
                        properties={"x": c * 16, "y": r * 16, "collected": False},
                        category="item",
                    )
                    graph.add_relation("player", "near" if
                        abs(sim.player_row - r) + abs(sim.player_col - c) < 4
                        else "far", f"heart_{heart_idx}")
                    heart_idx += 1

        # Chest and Exit
        for r in range(sim.GRID_H):
            for c in range(sim.GRID_W):
                if sim.grid[r, c] == Tile.CHEST:
                    graph.add_entity(
                        "chest",
                        properties={
                            "x": c * 16, "y": r * 16,
                            "opened": sim.chest_open,
                        },
                        category="door",
                    )
                    if sim.chest_open:
                        graph.add_relation("chest", "requires", "player")
                elif sim.grid[r, c] == Tile.EXIT:
                    graph.add_entity(
                        "exit",
                        properties={
                            "x": c * 16, "y": r * 16,
                            "locked": not sim.has_jewel,
                        },
                        category="door",
                    )
                    graph.add_relation("exit", "requires", "chest")

        # Enemies
        for i, e in enumerate(sim.enemies):
            if not e.alive:
                continue
            cat = _ENEMY_TO_GNN_CATEGORY.get(e.etype, "enemy")
            props = {
                "x": e.col * 16,
                "y": e.row * 16,
                "dangerous": e.is_dangerous,
                "alive": e.alive,
                "moving": e.etype in (EnemyType.ALMA, EnemyType.SKULL,
                                       EnemyType.DON_MEDUSA, EnemyType.ROCKY),
            }
            name = f"{e.etype.name.lower()}_{i}"
            graph.add_entity(name, properties=props, category=cat)

            # Threatening relations
            if e.is_dangerous:
                dist = abs(sim.player_row - e.row) + abs(sim.player_col - e.col)
                if dist < 3:
                    graph.add_relation(name, "threatens", "player")
                elif dist < 6:
                    graph.add_relation(name, "near", "player")

        # Pushable blocks
        block_idx = 0
        for r in range(sim.GRID_H):
            for c in range(sim.GRID_W):
                if sim.grid[r, c] == Tile.EMERALD:
                    graph.add_entity(
                        f"block_{block_idx}",
                        properties={
                            "x": c * 16, "y": r * 16,
                            "pushable": True,
                        },
                        category="block",
                    )
                    block_idx += 1

        # Auto-generate spatial relations
        graph.auto_spatial_relations(proximity_threshold=48.0)  # ~3 tiles

        return graph

    def grid_to_ram(self, sim: LoloSimulator) -> np.ndarray:
        """
        Convert grid state to fake 128-byte RAM for RAMSemanticMapper.

        Layout:
          0x00-0x01: player position (row, col)
          0x02: hearts collected
          0x03: hearts total
          0x04: magic shots
          0x05: flags (chest_open, has_jewel, alive)
          0x06: step count low byte
          0x07: facing direction
          0x10-0x3F: enemy data (8 enemies × 6 bytes each)
          0x40-0x7F: grid data (compressed: 2 cells per byte)
        """
        ram = np.zeros(128, dtype=np.uint8)

        # Player
        ram[0] = sim.player_row
        ram[1] = sim.player_col
        ram[2] = sim.hearts_collected
        ram[3] = sim.hearts_total
        ram[4] = sim.magic_shots
        ram[5] = (
            (int(sim.chest_open) << 0)
            | (int(sim.has_jewel) << 1)
            | (int(sim.alive) << 2)
            | (int(sim.won) << 3)
        )
        ram[6] = sim.step_count & 0xFF
        ram[7] = int(sim.facing)

        # Enemies (8 max, 6 bytes each at 0x10-0x3F)
        for i, e in enumerate(sim.enemies[:8]):
            base = 0x10 + i * 6
            ram[base] = e.etype
            ram[base + 1] = e.row
            ram[base + 2] = e.col
            ram[base + 3] = (
                (int(e.alive) << 0)
                | (int(e.is_egg) << 1)
                | (int(e.asleep) << 2)
                | (int(e.active) << 3)
            )
            ram[base + 4] = e.egg_timer & 0xFF
            ram[base + 5] = e.patrol_step

        # Grid (compressed: first 64 bytes of grid, 2 tiles per byte)
        flat_grid = sim.grid.flatten()
        for i in range(min(64, len(flat_grid) // 2)):
            lo = flat_grid[i * 2] & 0x0F
            hi = (flat_grid[i * 2 + 1] & 0x0F) << 4
            ram[0x40 + i] = lo | hi

        return ram

    # ── Save/Load passthrough ─────────────────────────────────────────

    def save_state(self) -> Optional[Dict[str, Any]]:
        if self._sim:
            return self._sim.save()
        return None

    def load_state(self, state: Dict[str, Any]) -> None:
        if self._sim:
            self._sim.load(state)

    def report(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "step_count": self._step_count,
            "n_obs": self._n_obs,
            "has_simulator": self._sim is not None,
        }
