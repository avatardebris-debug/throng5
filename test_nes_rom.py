"""Quick test: load an NES ROM and run random actions for 100 steps."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nes_py.nes_env import NESEnv
import numpy as np

ROM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roms", "nes")

# Find first available ROM
roms = [f for f in os.listdir(ROM_DIR) if f.endswith(".nes")]
if not roms:
    print("No .nes ROMs found in roms/nes/")
    sys.exit(1)

rom_path = os.path.join(ROM_DIR, roms[0])
print(f"Loading: {roms[0]}")
print(f"Path: {rom_path}")

# Create environment
env = NESEnv(rom_path)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Reset and run 100 random steps
obs = env.reset()
print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")

total_reward = 0.0
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        obs = env.reset()
        print(f"  Episode done at step {step}, reward so far: {total_reward:.1f}")

env.close()
print(f"\n100 steps completed!")
print(f"  Total reward: {total_reward:.1f}")
print(f"  Final obs shape: {obs.shape}")

# Now test with our adapter
print("\n--- Testing brain adapter ---")
from brain.environments.rom_adapter_factory import ROMAdapterFactory, EnvFingerprint, UniversalAdapter

fp = EnvFingerprint(
    name=roms[0],
    platform="nes",
    action_space_type="discrete",
    n_actions=env.action_space.n if hasattr(env.action_space, 'n') else 256,
    obs_shape=obs.shape,
    obs_type="pixels",
)
adapter = UniversalAdapter(fp, target_dim=84)

env2 = NESEnv(rom_path)
obs = env2.reset()
adapter.observe(obs)
features = adapter.make_features()
print(f"  NES pixels {obs.shape} -> brain features {features.shape}")
assert features.shape == (84,)
assert np.isfinite(features).all()

# Run 50 steps through adapter
for i in range(50):
    action = env2.action_space.sample()
    obs, reward, done, info = env2.step(action)
    adapter.observe(obs)
    features = adapter.make_features(action)
    if done:
        obs = env2.reset()

env2.close()
print(f"  50 steps through adapter: OK")
print(f"\nNES ROM LOADING TEST PASSED!")
