"""
run_lolo_rom.py — Test the brain on the real Lolo ROM (Linux + stable-retro only).

Uses the SARSA↔DQN bridge + WholeBrain to play real Adventures of Lolo.
Loads pretrained weights from simulator training and tests on actual game rooms.

Usage (on cloud VM):
  # First import the ROM
  python3 -m retro.import /path/to/roms/

  # Run with bridge
  python3 brain/games/lolo/run_lolo_rom.py --weights brain/games/lolo/dqn_weights.pt

  # Run with live SARSA solving (brain solves new rooms it hasn't seen)
  python3 brain/games/lolo/run_lolo_rom.py --live-solve

  # Benchmark: compare brain vs random
  python3 brain/games/lolo/run_lolo_rom.py --benchmark --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np


def run_brain_on_rom(
    env,
    brain,
    bridge,
    encoder,
    max_steps: int = 2000,
    verbose: bool = False,
) -> dict:
    """Run the brain on one room of the real ROM."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Get RAM state → compressed 84-dim features
        ram_state = env.get_ram_state()
        compressed = encoder.encode_from_ram(ram_state)

        # Bridge selects action (DQN or SARSA fallback)
        if bridge is not None:
            action, source = bridge.select_action_from_features(compressed)
        else:
            # Use brain directly
            result = brain.step(
                obs=compressed,
                prev_action=0,
                reward=total_reward,
                done=False,
            )
            action = result.get("action", 0) % 6
            source = "brain"

        # Step ROM
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose and step % 50 == 0:
            print(f"    step={step} reward={total_reward:.1f} "
                  f"hearts={info['hearts']}/{info['hearts_total']} "
                  f"source={source}", flush=True)

        # Observe for bridge learning
        if bridge is not None:
            bridge.observe_from_features(compressed, action, reward, done)

        if done or truncated:
            break

    return {
        "won": env.won,
        "steps": steps,
        "reward": total_reward,
        "hearts": env.hearts_collected,
        "hearts_total": env.hearts_total,
    }


def run_random_on_rom(env, max_steps: int = 2000) -> dict:
    """Random baseline on real ROM."""
    env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = np.random.randint(6)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    return {
        "won": env.won,
        "steps": step + 1,
        "reward": total_reward,
        "hearts": env.hearts_collected,
        "hearts_total": env.hearts_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Run brain on real Lolo ROM")
    parser.add_argument("--weights", default="brain/games/lolo/dqn_weights.pt")
    parser.add_argument("--sarsa", default="brain/games/lolo/sarsa_qtable.npy")
    parser.add_argument("--game", default="AdventuresOfLolo-Nes",
                        help="stable-retro game name")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--benchmark", action="store_true",
                        help="Also run random baseline for comparison")
    parser.add_argument("--live-solve", action="store_true",
                        help="Use SARSA live-solving for unknown rooms")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  LOLO ROM TEST — Real NES Game via stable-retro")
    print("=" * 60)

    # ── Import checks ──
    try:
        from brain.games.lolo.lolo_rom_env import LoloROMEnv
    except ImportError as e:
        print(f"  ❌ Cannot load ROM environment: {e}")
        print(f"  Install: pip install stable-retro")
        print(f"  Import ROM: python -m retro.import /path/to/roms/")
        sys.exit(1)

    from brain.games.lolo.lolo_compressed_state import LoloCompressedState
    encoder = LoloCompressedState()

    # ── Load environment ──
    print(f"\n  Loading ROM: {args.game}...")
    try:
        env = LoloROMEnv(game=args.game)
        print(f"  ✅ ROM loaded")
    except Exception as e:
        print(f"  ❌ Failed to load ROM: {e}")
        print(f"  Make sure the ROM is imported with: python -m retro.import /path/to/roms/")
        sys.exit(1)

    # ── Load bridge ──
    bridge = None
    if os.path.exists(args.weights) or os.path.exists(args.sarsa):
        from brain.games.lolo.sarsa_dqn_bridge import SarsaDQNBridge
        bridge = SarsaDQNBridge(n_actions=6)
        if os.path.exists(args.sarsa):
            bridge.load_sarsa(args.sarsa)
        if os.path.exists(args.weights):
            bridge.load_dqn(args.weights)
        print(f"  ✅ Bridge loaded (SARSA: {bridge.has_sarsa()}, DQN: {bridge.has_dqn()})")
    else:
        print(f"  ⚠ No weights found, running brain without pretrained knowledge")

    # ── Build brain ──
    from brain.orchestrator import WholeBrain
    brain = WholeBrain(n_features=84, n_actions=6,
                       use_cnn=False, use_fft=False, use_torch=True)

    # ── Run episodes ──
    print(f"\n  Running {args.episodes} episodes (max {args.max_steps} steps)...")

    brain_wins = 0
    brain_hearts = 0
    total_hearts = 0

    for ep in range(args.episodes):
        result = run_brain_on_rom(
            env, brain, bridge, encoder,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )

        status = "✅ WON" if result["won"] else "❌"
        print(f"  Episode {ep+1}: {status}  steps={result['steps']:4d}  "
              f"hearts={result['hearts']}/{result['hearts_total']}  "
              f"reward={result['reward']:.1f}", flush=True)

        if result["won"]:
            brain_wins += 1
        brain_hearts += result["hearts"]
        total_hearts += result["hearts_total"]

    brain_rate = brain_wins / max(args.episodes, 1)
    heart_rate = brain_hearts / max(total_hearts, 1)
    print(f"\n  Brain: {brain_wins}/{args.episodes} won ({brain_rate:.0%}), "
          f"hearts: {brain_hearts}/{total_hearts} ({heart_rate:.0%})")

    # ── Random baseline ──
    if args.benchmark:
        print(f"\n  Running random baseline ({args.episodes} episodes)...")
        random_wins = 0
        for ep in range(args.episodes):
            result = run_random_on_rom(env, max_steps=args.max_steps)
            if result["won"]:
                random_wins += 1

        random_rate = random_wins / max(args.episodes, 1)
        print(f"  Random: {random_wins}/{args.episodes} ({random_rate:.0%})")
        print(f"  Brain advantage: {brain_rate - random_rate:+.0%}")

    # ── Bridge report ──
    if bridge is not None:
        print(f"\n  Bridge report: {bridge.report()}")

    env.close()


if __name__ == "__main__":
    main()
