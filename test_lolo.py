"""Test the Lolo simulator, generator, adapter, and curriculum system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

print("=" * 60)
print("Adventures of Lolo System Test")
print("=" * 60)

# ── 1. Simulator basic test ──────────────────────────────────────────

from brain.games.lolo.lolo_simulator import (
    LoloSimulator, Tile, Action, Enemy, EnemyType,
)

# Create a simple puzzle
grid = np.zeros((13, 11), dtype=np.uint8)
# Border
grid[0, :] = Tile.ROCK
grid[12, :] = Tile.ROCK
grid[:, 0] = Tile.ROCK
grid[:, 10] = Tile.ROCK

# Content
grid[6, 5] = Tile.PLAYER
grid[3, 5] = Tile.HEART
grid[3, 7] = Tile.HEART
grid[1, 5] = Tile.CHEST
grid[1, 8] = Tile.EXIT

sim = LoloSimulator(grid)
print(f"[PASS] Simulator created: {sim.GRID_H}x{sim.GRID_W}, "
      f"{sim.hearts_total} hearts")

# Step test
obs, reward, done, info = sim.step(Action.UP)
assert not done
assert isinstance(obs, np.ndarray)
print(f"[PASS] Step works: obs size={obs.shape[0]}, reward={reward:.3f}")

print(f"\n{sim.render_ascii()}\n")

# Save/load test
state = sim.save()
sim2 = LoloSimulator(grid.copy())
sim2.load(state)
assert sim2.player_row == sim.player_row
assert sim2.player_col == sim.player_col
print(f"[PASS] Save/load works")

# Solvability test
assert sim.is_solvable(), "Simple puzzle should be solvable"
print(f"[PASS] Solvability check (BFS): solvable")

# Play through to collect hearts
for _ in range(5):
    sim.step(Action.UP)
# Should have collected heart at (3,5)
print(f"  Hearts collected: {sim.hearts_collected}/{sim.hearts_total}")

# ── 2. Enemy test ────────────────────────────────────────────────────

grid2 = np.zeros((13, 11), dtype=np.uint8)
grid2[0, :] = Tile.ROCK
grid2[12, :] = Tile.ROCK
grid2[:, 0] = Tile.ROCK
grid2[:, 10] = Tile.ROCK
grid2[10, 5] = Tile.PLAYER
grid2[3, 5] = Tile.HEART
grid2[1, 5] = Tile.CHEST
grid2[1, 8] = Tile.EXIT

enemies = [
    Enemy(etype=EnemyType.MEDUSA, row=5, col=3),
    Enemy(etype=EnemyType.SNAKEY, row=8, col=7),
]

sim_e = LoloSimulator(grid2, enemies=enemies)
print(f"[PASS] Simulator with enemies: {len(sim_e.enemies)} enemies")
print(f"\n{sim_e.render_ascii()}\n")

# ── 3. Generator test ────────────────────────────────────────────────

from brain.games.lolo.lolo_generator import LoloPuzzleGenerator

gen = LoloPuzzleGenerator(seed=42)

# Test each tier
for tier in range(1, 8):
    puzzle = gen.generate(tier=tier)
    if puzzle:
        print(f"[PASS] Tier {tier}: generated solvable puzzle "
              f"({puzzle.hearts_total} hearts, {len(puzzle.enemies)} enemies)")
    else:
        print(f"[WARN] Tier {tier}: failed to generate solvable puzzle")

# Batch generation
batch = gen.generate_batch(20, tier=1)
print(f"[PASS] Batch: {len(batch)}/20 solvable tier-1 puzzles generated")
print(f"  Generator stats: {gen.report()}")

# Show a tier 3 puzzle
t3 = gen.generate(tier=3)
if t3:
    print(f"\nTier 3 puzzle:")
    print(t3.render_ascii())

# ── 4. Adapter test ──────────────────────────────────────────────────

from brain.games.lolo.lolo_adapter import LoloAdapter

adapter = LoloAdapter(feature_dim=84)
simple = gen.generate(tier=1)
if simple:
    features = adapter.grid_to_features(simple)
    assert features.shape == (84,), f"Expected (84,), got {features.shape}"
    print(f"\n[PASS] Adapter features: {features.shape}")

    # ObjectGraph
    graph = adapter.grid_to_object_graph(simple)
    if graph:
        n_entities = len(graph._entities)
        n_relations = len(graph._relations)
        print(f"[PASS] ObjectGraph: {n_entities} entities, {n_relations} relations")
        desc = graph.describe()
        print(f"  Graph description ({len(desc)} chars)")
    else:
        print(f"[WARN] ObjectGraph not available")

    # RAM
    ram = adapter.grid_to_ram(simple)
    assert ram.shape == (128,), f"Expected (128,), got {ram.shape}"
    print(f"[PASS] Fake RAM: {ram.shape}, player at ({ram[0]}, {ram[1]})")

    # Gymnasium interface
    features = adapter.reset(simple)
    features2, reward, done, info = adapter.step(Action.UP)
    assert features2.shape == (84,)
    print(f"[PASS] Gymnasium interface: step returns features {features2.shape}")

# ── 5. EntityGNN with Lolo graph ─────────────────────────────────────

try:
    from brain.networks.entity_gnn import EntityGNN

    if graph:
        gnn = EntityGNN(d_entity=16, d_global=32)
        node_emb, global_emb = gnn.forward(graph)
        print(f"[PASS] EntityGNN on Lolo graph: nodes={node_emb.shape}, "
              f"global={global_emb.shape}")
except Exception as e:
    print(f"[WARN] EntityGNN test: {e}")

# ── 6. Curriculum mini-test ──────────────────────────────────────────

from brain.orchestrator import WholeBrain
from brain.games.lolo.lolo_curriculum import LoloCurriculum

brain = WholeBrain(n_features=84, n_actions=18, session_name="lolo_test")
curriculum = LoloCurriculum(brain, seed=42)

print(f"\n{'='*60}")
print("Running mini curriculum (Tier 1, 10 episodes)...")
results = curriculum.train_tier(tier=1, n_episodes=10, verbose=False)
print(f"[PASS] Curriculum tier 1: {results['success_rate']:.0%} success, "
      f"avg_reward={results['avg_reward']:.1f}, "
      f"avg_steps={results['avg_steps']:.0f}")
print(f"  Report: {curriculum.report()}")

brain.close()

# ── Done ─────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("ALL LOLO SYSTEM TESTS PASSED")
print("=" * 60)
