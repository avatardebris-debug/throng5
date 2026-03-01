"""
run_tiered_gan.py — Auto-climbing GAN training: Tier 1 → 7.

Strategy per tier:
  1. Seed with 200 random puzzles (decaying schedule: 200→5 tries)
  2. Pretrain generator from solved puzzle bank
  3. Run GAN pipeline (100 puzzles per round)
  4. If solvability > threshold, advance to next tier
  5. Carry DQN weights and solved bank forward

DQN generalizes across puzzles via neural network on 84-dim compressed state.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Any

sys.path.insert(0, ".")

import numpy as np

from brain.games.lolo.lolo_gan import LoloGAN
from brain.games.lolo.lolo_gan_trainer import GanTrainingLoop


# ── Configuration ─────────────────────────────────────────────────────

USE_DQN = True  # Set False to use tabular SARSA instead

TIER_NAMES = {
    1: "Basic Navigation",
    2: "Pushable Blocks",
    3: "Line of Sight (Medusa)",
    4: "Enemy Manipulation (Shoot→Egg)",
    5: "Active Threats (Moving Enemies)",
    6: "Water Mechanics (Egg→Bridge)",
    7: "Endgame (All Enemies)",
}

# Per-tier training parameters
TIER_PARAMS = {
    1: {"seed_n": 200, "rounds": 3, "puzzles_per_round": 100, "advance_threshold": 0.30},
    2: {"seed_n": 200, "rounds": 3, "puzzles_per_round": 100, "advance_threshold": 0.25},
    3: {"seed_n": 200, "rounds": 4, "puzzles_per_round": 100, "advance_threshold": 0.20},
    4: {"seed_n": 200, "rounds": 4, "puzzles_per_round": 100, "advance_threshold": 0.15},
    5: {"seed_n": 250, "rounds": 5, "puzzles_per_round": 100, "advance_threshold": 0.10},
    6: {"seed_n": 250, "rounds": 5, "puzzles_per_round": 100, "advance_threshold": 0.10},
    7: {"seed_n": 300, "rounds": 6, "puzzles_per_round": 100, "advance_threshold": 0.05},
}


def p(msg):
    print(msg, flush=True)


def learner_size(trainer) -> str:
    """Get learner size info (works for both SARSA and DQN)."""
    r = trainer.sarsa.report()
    if r.get("type") == "LoloDQNLearner":
        return f"steps={r['total_steps']} learns={r['total_learns']} loss={r.get('avg_loss', 0):.4f}"
    else:
        return f"Q={r.get('q_table_size', len(trainer.sarsa.q_table))}"


def run_tier(
    trainer: GanTrainingLoop,
    tier: int,
    params: dict,
) -> Dict[str, Any]:
    """Run full GAN training for a single tier."""
    tier_name = TIER_NAMES.get(tier, f"Tier {tier}")
    p(f"\n{'='*70}")
    p(f"  TIER {tier}: {tier_name}")
    p(f"{'='*70}")

    trainer.tier = tier
    t0 = time.time()

    # Phase 1: Seed with random puzzles
    p(f"\n  Phase 1: Seeding {params['seed_n']} random puzzles...")
    seed_result = trainer.seed_with_random(n=params["seed_n"], tier=tier)
    solved_pct = seed_result["solved"] / max(seed_result["seeded"], 1)
    p(f"  Seed done: {seed_result['solved']}/{seed_result['seeded']} solved "
      f"({solved_pct:.0%}), {learner_size(trainer)}")

    # Phase 2: GAN rounds
    round_results = []
    best_solvability = 0.0

    for rnd in range(params["rounds"]):
        p(f"\n  Round {rnd+1}/{params['rounds']}:")

        # Reset per-round counters (but keep banks)
        trainer._total_generated = 0
        trainer.UNSOLVED_BATCH_SIZE = 30
        trainer.STAGE3_EPISODES = 50   # Lighter PPO fallback

        result = trainer.run(n_puzzles=params["puzzles_per_round"], verbose=True)

        total = result["generated"]
        solved = result["solved"]
        solvability = solved / max(total, 1)
        round_results.append({
            "round": rnd + 1,
            "generated": total,
            "solved": solved,
            "solvability": round(solvability, 3),
            "unsolvable": result["unsolvable"],
            "gan_steps": result["gan_train_steps"],
        })

        best_solvability = max(best_solvability, solvability)

        p(f"    → Solved {solved}/{total} ({solvability:.0%}), "
          f"best={best_solvability:.0%}, {learner_size(trainer)}")

        # Check advance threshold
        if solvability >= params["advance_threshold"]:
            p(f"    ✅ Threshold {params['advance_threshold']:.0%} reached!")
            break

    elapsed = time.time() - t0

    tier_result = {
        "tier": tier,
        "name": tier_name,
        "elapsed": round(elapsed, 1),
        "best_solvability": round(best_solvability, 3),
        "advanced": best_solvability >= params["advance_threshold"],
        "rounds": round_results,
        "learner_report": trainer.sarsa.report(),
        "solved_bank_size": len(trainer.gan.solved_bank),
        "gan_report": trainer.gan.report(),
    }

    p(f"\n  Tier {tier} complete in {elapsed:.0f}s — "
      f"best solvability: {best_solvability:.0%}, "
      f"{learner_size(trainer)}, "
      f"solved bank: {len(trainer.gan.solved_bank)}")

    return tier_result


def main():
    p("=" * 70)
    mode = "DQN" if USE_DQN else "SARSA"
    p(f"  LOLO GAN — TIERED TRAINING (1 → 7) [{mode}]")
    p("=" * 70)

    # Shared GAN and trainer — knowledge carries forward
    gan = LoloGAN(z_dim=32, lr=0.0005)
    trainer = GanTrainingLoop(gan=gan, tier=1, use_dqn=USE_DQN)

    all_results = []
    overall_t0 = time.time()

    for tier in range(1, 8):
        params = TIER_PARAMS[tier]
        tier_result = run_tier(trainer, tier, params)
        all_results.append(tier_result)

        # Save checkpoint after each tier
        checkpoint = {
            "learner_type": mode,
            "completed_tiers": [r["tier"] for r in all_results],
            "results": all_results,
            "total_solved_bank": len(trainer.gan.solved_bank),
            "total_elapsed": round(time.time() - overall_t0, 1),
        }
        checkpoint_path = os.path.join("brain", "games", "lolo", "gan_checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        p(f"\n  📁 Checkpoint saved: {checkpoint_path}")

        # Save DQN weights after each tier
        if USE_DQN:
            dqn_path = os.path.join("brain", "games", "lolo", "dqn_weights.pt")
            trainer.sarsa.save(dqn_path)
            p(f"  💾 DQN weights saved: {dqn_path}")

        if not tier_result["advanced"]:
            p(f"\n  ⚠️ Tier {tier} not advanced — stopping here.")
            p(f"     Best solvability: {tier_result['best_solvability']:.0%}")
            p(f"     Threshold was: {params['advance_threshold']:.0%}")
            break

    # ── Final Report ──
    total_time = time.time() - overall_t0
    p(f"\n{'='*70}")
    p(f"  FINAL REPORT — {total_time:.0f}s total [{mode}]")
    p(f"{'='*70}")
    p(f"\n  {'Tier':<6} {'Name':<30} {'Best%':<8} {'Advanced'}")
    p(f"  {'─'*6} {'─'*30} {'─'*8} {'─'*8}")
    for r in all_results:
        p(f"  {r['tier']:<6} {r['name']:<30} {r['best_solvability']:.0%}{'':>4} "
          f"{'✅' if r['advanced'] else '❌'}")

    p(f"\n  Learner: {trainer.sarsa.report()}")
    p(f"  Solved bank: {len(trainer.gan.solved_bank)}")
    p(f"  GAN report:  {trainer.gan.report()}")


if __name__ == "__main__":
    main()
