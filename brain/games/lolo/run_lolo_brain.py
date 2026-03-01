"""
run_lolo_brain.py — Run the full WholeBrain on Lolo puzzles.

Tests the complete brain pipeline:
  Sensory → Basal Ganglia (with DQN habit) → Amygdala → Hippocampus →
  Striatum → Motor Cortex → Lolo Action

Uses the Lolo simulator (identical mechanics to NES ROM).
When stable-retro works on Windows or cloud Linux, swap simulator
for NES env — the compressed state encoder is the same either way.

Usage:
  python brain/games/lolo/run_lolo_brain.py
  python brain/games/lolo/run_lolo_brain.py --weights brain/games/lolo/dqn_weights.pt
  python brain/games/lolo/run_lolo_brain.py --no-habit   # brain without pretrained DQN
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from brain.games.lolo.lolo_compressed_state import LoloCompressedState
from brain.games.lolo.lolo_generator import LoloPuzzleGenerator
from brain.games.lolo.lolo_simulator import LoloSimulator


def run_with_brain(sim: LoloSimulator, brain, encoder: LoloCompressedState,
                   max_steps: int = 500, verbose: bool = False) -> dict:
    """Run one puzzle with the full brain."""
    initial_state = sim.save()  # save fresh state for reference
    total_reward = 0.0
    steps = 0

    obs = sim.get_obs()
    prev_action = 0

    for step in range(max_steps):
        # Get compressed features for basal ganglia
        features = encoder.encode_from_sim(sim)

        # Brain step — pass sim for habit encoding
        result = brain.step(
            obs=features,       # 84-dim compressed state as observation
            prev_action=prev_action,
            reward=0.0 if step == 0 else total_reward,
            done=False,
        )

        action = result.get("action", 0)

        # Map brain action (may be 0-17) to Lolo action (0-5)
        lolo_action = action % 6  # UP, DOWN, LEFT, RIGHT, SHOOT, WAIT

        # Step simulator
        obs, reward, done, info = sim.step(lolo_action)
        total_reward += reward
        prev_action = lolo_action
        steps += 1

        if verbose and step % 50 == 0:
            print(f"    step={step} reward={total_reward:.1f} "
                  f"hearts={sim.hearts_collected}/{sim.hearts_total} "
                  f"pos=({sim.player_row},{sim.player_col})", flush=True)

        if done:
            break

    return {
        "won": sim.won,
        "steps": steps,
        "reward": total_reward,
        "hearts": sim.hearts_collected,
        "hearts_total": sim.hearts_total,
    }


def run_without_brain(sim: LoloSimulator, encoder: LoloCompressedState,
                       max_steps: int = 500) -> dict:
    """Run one puzzle with random actions (baseline)."""
    total_reward = 0.0

    for step in range(max_steps):
        action = np.random.randint(6)
        obs, reward, done, info = sim.step(action)
        total_reward += reward
        if done:
            break

    return {
        "won": sim.won,
        "steps": step + 1,
        "reward": total_reward,
        "hearts": sim.hearts_collected,
        "hearts_total": sim.hearts_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Run full brain on Lolo puzzles")
    parser.add_argument("--weights", default="brain/games/lolo/dqn_weights.pt",
                        help="Path to DQN weights (.pt)")
    parser.add_argument("--sarsa", default="brain/games/lolo/sarsa_qtable.npy",
                        help="Path to SARSA Q-table (.npy)")
    parser.add_argument("--no-habit", action="store_true",
                        help="Run brain WITHOUT any pretrained knowledge (baseline)")
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3],
                        help="Which tiers to test")
    parser.add_argument("--puzzles", type=int, default=20,
                        help="Puzzles per tier")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per puzzle")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  LOLO BRAIN TEST — Full WholeBrain on Simulator")
    print("=" * 60)

    encoder = LoloCompressedState()
    gen = LoloPuzzleGenerator(seed=42)

    # ── Build brain ──
    from brain.orchestrator import WholeBrain

    brain = WholeBrain(
        n_features=84,
        n_actions=6,   # Lolo has 6 actions
        use_cnn=False,  # We use compressed state, not pixels
        use_fft=False,
        use_torch=True,
    )

    # Load knowledge (try bridge first, then standalone DQN)
    mode = "NONE"
    if not args.no_habit:
        # Try dual-process bridge (SARSA + DQN)
        has_sarsa = os.path.exists(args.sarsa)
        has_dqn = os.path.exists(args.weights)

        if has_sarsa or has_dqn:
            bridge_ok = brain.basal_ganglia.load_bridge(
                sarsa_path=args.sarsa, dqn_path=args.weights
            )
            if bridge_ok:
                mode = "BRIDGE (SARSA+DQN)"
                print(f"  ✅ Dual-process bridge loaded")
                if has_sarsa:
                    print(f"     SARSA: {args.sarsa}")
                if has_dqn:
                    print(f"     DQN:   {args.weights}")
            elif has_dqn:
                # Fallback to standalone DQN
                habit_loaded = brain.basal_ganglia.load_habit_weights(
                    args.weights, game="lolo"
                )
                if habit_loaded:
                    mode = "DQN-only"
                    print(f"  ✅ DQN habit loaded: {args.weights}")

        if mode == "NONE":
            print(f"  ⚠️ No weights found")
            print("     Run 'python brain/games/lolo/distill_sarsa_to_dqn.py' first")
            print("     Or use --no-habit for baseline comparison")
    else:
        print("  🚫 Running WITHOUT pretrained knowledge (baseline)")

    print(f"\n  Tiers: {args.tiers}")
    print(f"  Puzzles per tier: {args.puzzles}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Mode: {mode}")

    # ── Run tests ──
    all_results = {}

    for tier in args.tiers:
        print(f"\n{'─'*60}")
        print(f"  Tier {tier}")
        print(f"{'─'*60}")

        tier_wins = 0
        tier_hearts = 0
        tier_total_hearts = 0

        for i in range(args.puzzles):
            sim = gen.generate(tier=tier, max_attempts=500)
            if sim is None:
                continue

            # Run with brain
            result = run_with_brain(
                sim, brain, encoder,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )

            status = "✅ WON" if result["won"] else "❌"
            hearts = f"{result['hearts']}/{result['hearts_total']}"
            print(f"  Puzzle {i+1:3d}: {status}  steps={result['steps']:4d}  "
                  f"hearts={hearts}  reward={result['reward']:.1f}",
                  flush=True)

            if result["won"]:
                tier_wins += 1
            tier_hearts += result["hearts"]
            tier_total_hearts += result["hearts_total"]

        win_rate = tier_wins / max(args.puzzles, 1)
        heart_rate = tier_hearts / max(tier_total_hearts, 1)
        all_results[tier] = {
            "wins": tier_wins,
            "puzzles": args.puzzles,
            "win_rate": win_rate,
            "heart_rate": heart_rate,
        }

        print(f"\n  Tier {tier}: {tier_wins}/{args.puzzles} won ({win_rate:.0%}), "
              f"hearts: {tier_hearts}/{tier_total_hearts} ({heart_rate:.0%})")

    # ── Also run random baseline for comparison ──
    print(f"\n{'─'*60}")
    print(f"  RANDOM BASELINE (for comparison)")
    print(f"{'─'*60}")

    for tier in args.tiers:
        random_wins = 0
        for i in range(args.puzzles):
            sim = gen.generate(tier=tier, max_attempts=500)
            if sim is None:
                continue
            result = run_without_brain(sim, encoder, max_steps=args.max_steps)
            if result["won"]:
                random_wins += 1

        rand_rate = random_wins / max(args.puzzles, 1)
        print(f"  Tier {tier} random: {random_wins}/{args.puzzles} ({rand_rate:.0%})")
        all_results[tier]["random_wins"] = random_wins
        all_results[tier]["random_rate"] = rand_rate

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Brain vs Random")
    print(f"{'='*60}")
    print(f"  {'Tier':<6} {'Brain':>10} {'Random':>10} {'Δ':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8}")
    for tier, r in all_results.items():
        delta = r['win_rate'] - r.get('random_rate', 0)
        print(f"  {tier:<6} {r['win_rate']:>9.0%} {r.get('random_rate',0):>9.0%} "
              f"{'+' if delta >= 0 else ''}{delta:>6.0%}")

    print(f"\n  Brain report: {brain.basal_ganglia.report()}")


if __name__ == "__main__":
    main()
