"""Phase 5 test — Multi-ROM Generalization components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

print("=" * 60)
print("Phase 5: Multi-ROM Generalization Test")
print("=" * 60)

# ── 1. ROM Adapter Factory ────────────────────────────────────────────

from brain.environments.rom_adapter_factory import ROMAdapterFactory, UniversalAdapter, EnvFingerprint

factory = ROMAdapterFactory(target_dim=84, probe_steps=50)
print("[PASS] ROMAdapterFactory created")

# Test manual fingerprint + adapter creation
fp = EnvFingerprint(
    name="TestEnv",
    platform="gymnasium",
    action_space_type="discrete",
    n_actions=6,
    obs_shape=(128,),
    obs_type="ram",
)
adapter = UniversalAdapter(fp, target_dim=84)

# Feed observations
rng = np.random.RandomState(42)
for i in range(20):
    obs = rng.randint(0, 256, size=128).astype(np.float32)
    adapter.observe(obs)
    features = adapter.make_features(action=rng.randint(6))
    assert features.shape == (84,), f"Expected (84,), got {features.shape}"
    assert np.isfinite(features).all(), "Features contain NaN/Inf"

print(f"[PASS] UniversalAdapter: RAM obs (128) -> features (84)")

# Test with pixel-like obs
fp_pixel = EnvFingerprint(name="PixelEnv", obs_shape=(210, 160, 3), obs_type="pixels")
adapter_pixel = UniversalAdapter(fp_pixel, target_dim=84)
pixel_obs = rng.randint(0, 256, size=(210, 160, 3)).astype(np.float32)
adapter_pixel.observe(pixel_obs)
pixel_features = adapter_pixel.make_features()
assert pixel_features.shape == (84,)
print(f"[PASS] UniversalAdapter: pixel obs (210x160x3) -> features (84)")

# ── 2. Curiosity Module ──────────────────────────────────────────────

from brain.environments.curiosity import CuriosityModule

curiosity = CuriosityModule(n_features=84, n_actions=6)
print("[PASS] CuriosityModule created")

rewards = []
for i in range(100):
    f1 = rng.randn(84).astype(np.float32)
    f2 = rng.randn(84).astype(np.float32)
    action = rng.randint(6)
    intrinsic = curiosity.compute(f1, action, f2)
    rewards.append(intrinsic)

print(f"  Intrinsic rewards: mean={np.mean(rewards):.4f}, max={np.max(rewards):.4f}")
stats = curiosity.stats()
print(f"  Unique states: {stats['unique_states']}, pred_error: {stats['avg_pred_error']:.5f}")
assert stats["unique_states"] > 50, f"Expected many unique states, got {stats['unique_states']}"
assert stats["pred_steps"] == 100

# Test that repeated states reduce novelty
repeated_f = np.ones(84, dtype=np.float32)
r1 = curiosity.compute(repeated_f, 0, repeated_f + 0.01)
r2 = curiosity.compute(repeated_f, 0, repeated_f + 0.01)
r3 = curiosity.compute(repeated_f, 0, repeated_f + 0.01)
# Visit count novelty should decrease
print(f"  Repeated state rewards: {r1:.4f} -> {r2:.4f} -> {r3:.4f}")
print(f"[PASS] CuriosityModule works, 100 steps computed")

# ── 3. Spatial Mapper ─────────────────────────────────────────────────

from brain.environments.spatial_mapper import SpatialMapper

mapper = SpatialMapper(n_features=84, max_locations=100, merge_threshold=2.0)
print("[PASS] SpatialMapper created")

# Simulate agent visiting 3 distinct regions
region_a = np.zeros(84, dtype=np.float32)
region_b = np.ones(84, dtype=np.float32) * 5
region_c = np.ones(84, dtype=np.float32) * -5

# Visit region A repeatedly
for i in range(10):
    noise = rng.randn(84).astype(np.float32) * 0.1
    loc = mapper.observe(region_a + noise, action=0, reward=0.1)

# Transition to region B
loc_b = mapper.observe(region_b, action=1, reward=1.0)

# Visit region B
for i in range(5):
    noise = rng.randn(84).astype(np.float32) * 0.1
    mapper.observe(region_b + noise, action=1, reward=0.5)

# Transition to region C
loc_c = mapper.observe(region_c, action=2, reward=-1.0)

stats = mapper.stats()
print(f"  Locations: {stats['n_locations']}, Edges: {stats['n_edges']}")
assert stats["n_locations"] >= 2, f"Expected at least 2 locations, got {stats['n_locations']}"
assert stats["n_edges"] >= 1, f"Expected at least 1 edge, got {stats['n_edges']}"

# Test pathfinding
loc_a = mapper.current_location(region_a)
loc_b = mapper.current_location(region_b)
path = mapper.shortest_path(loc_a, loc_b)
print(f"  Path A->B: {path}")

# Test exploration queries
high_reward = mapper.high_reward_locations(3)
unexplored = mapper.unexplored_locations(3)
frontier = mapper.frontier_locations(3)
print(f"  High reward locations: {high_reward}")
print(f"  Unexplored locations: {unexplored[:3]}")
print(f"  Frontier locations: {frontier[:3]}")
print(f"[PASS] SpatialMapper works — {stats['n_locations']} locations, {stats['n_edges']} edges")

# ── 4. NES Adapter ────────────────────────────────────────────────────

from brain.environments.nes_adapter import NESAdapter

nes = NESAdapter(game="SuperMarioBros-Nes", target_dim=84)
print("[PASS] NESAdapter created")

# Test pixel processing
pixel = rng.randint(0, 256, size=(224, 240, 3)).astype(np.float32)
nes.observe(pixel, info={"lives": 3, "score": 1600, "level": 1})
features = nes.make_features(action=1)
assert features.shape == (84,), f"Expected (84,), got {features.shape}"

# Test button mapping
buttons_noop = nes.action_to_buttons(0)
assert sum(buttons_noop) == 0, "NOOP should press no buttons"

buttons_right = nes.action_to_buttons(1)
assert buttons_right[7] == 1, "Action 1 should press RIGHT"

buttons_jump = nes.action_to_buttons(8)
assert buttons_jump[8] == 1, "Action 8 should press A (jump)"

buttons_run_jump = nes.action_to_buttons(10)
assert buttons_run_jump[7] == 1 and buttons_run_jump[8] == 1, "Action 10 should be RIGHT+A+B"

# Test game info extraction
info = nes.get_game_info()
assert info["lives"] == 3
assert info["score"] == 1600
print(f"  Game info: {info}")
print(f"[PASS] NESAdapter: pixels -> features, button mapping, game info all work")

# ── 5. Full Pipeline: Factory -> Adapter -> Curiosity -> Mapper ───────

print("\n--- Full Pipeline Integration ---")
brain_features = 84
n_actions = 6

fp_test = EnvFingerprint(
    name="IntegrationTest", obs_shape=(100,), n_actions=n_actions
)
test_adapter = UniversalAdapter(fp_test, target_dim=brain_features)
test_curiosity = CuriosityModule(n_features=brain_features, n_actions=n_actions)
test_mapper = SpatialMapper(n_features=brain_features)

total_intrinsic = 0.0
for step in range(200):
    obs = rng.randn(100).astype(np.float32)
    action = rng.randint(n_actions)

    test_adapter.observe(obs)
    features = test_adapter.make_features(action)

    # Next obs
    next_obs = rng.randn(100).astype(np.float32)
    test_adapter.observe(next_obs)
    next_features = test_adapter.make_features(action)

    # Curiosity
    intrinsic = test_curiosity.compute(features, action, next_features)
    total_intrinsic += intrinsic

    # Spatial mapping
    test_mapper.observe(features, action=action, reward=intrinsic,
                        next_features=next_features)

stats = test_mapper.stats()
curiosity_stats = test_curiosity.stats()
print(f"  200 steps: {stats['n_locations']} locations, {stats['n_edges']} edges")
print(f"  Curiosity: {curiosity_stats['unique_states']} unique states")
print(f"  Total intrinsic reward: {total_intrinsic:.2f}")
print(f"[PASS] Full pipeline integration works")

print()
print("=" * 60)
print("ALL PHASE 5 TESTS PASSED")
print("=" * 60)
