"""
train_game.py — One-command cloud trainer for any game.

Spin up a VM, run this, walk away. Handles:
  - Game-specific SARSA+GAN training
  - DQN distillation from SARSA
  - Auto-checkpoint every N minutes
  - Weight push to GCS when done

Usage:
  python cloud/train_game.py --game lolo --hours 1
  python cloud/train_game.py --game lolo --hours 0.5 --checkpoint-interval 5
  python cloud/train_game.py --game montezuma --hours 2

Supported games: lolo (more coming)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, ".")

import numpy as np

# ── Game Configs ──────────────────────────────────────────────────────

GAME_CONFIGS = {
    "lolo": {
        "module": "brain.games.lolo",
        "n_actions": 6,
        "state_dim": 84,
        "tiers": list(range(1, 8)),
        "seed_per_tier": {1: 200, 2: 200, 3: 200, 4: 200, 5: 250, 6: 250, 7: 300},
        "gan_puzzles_per_round": 100,
        "episodes_per_tier": 100,
    },
}


def train_lolo(hours: float, checkpoint_min: int, output_dir: str, verbose: bool):
    """Train Lolo: SARSA through tiers, GAN rounds, then distill → DQN."""
    from brain.games.lolo.lolo_gan import LoloGAN
    from brain.games.lolo.lolo_gan_trainer import GanTrainingLoop
    from brain.games.lolo.lolo_dqn_learner import LoloDQNLearner
    from brain.games.lolo.distill_sarsa_to_dqn import distill, verify_distillation

    t_start = time.time()
    t_end = t_start + hours * 3600
    t_next_checkpoint = t_start + checkpoint_min * 60

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "dqn_weights.pt")
    sarsa_path = os.path.join(output_dir, "sarsa_qtable.npy")
    stats_path = os.path.join(output_dir, "training_stats.json")

    gan = LoloGAN(z_dim=32, lr=0.0005)
    trainer = GanTrainingLoop(gan=gan, tier=1, use_dqn=False)

    from brain.games.lolo.run_tiered_gan import TIER_PARAMS

    stats = {
        "game": "lolo",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiers_completed": [],
        "total_q_states": 0,
        "checkpoints_saved": 0,
        "rounds": [],
    }

    tier_idx = 0
    round_num = 0

    # ── Phase 1: SARSA+GAN Training ──
    print(f"\n{'='*60}")
    print(f"  LOLO CLOUD TRAINER — {hours}h budget, checkpoint every {checkpoint_min}m")
    print(f"{'='*60}")

    while time.time() < t_end:
        tier = (tier_idx % 7) + 1
        params = TIER_PARAMS[tier]
        trainer.tier = tier
        round_num += 1

        elapsed_min = (time.time() - t_start) / 60
        remaining_min = (t_end - time.time()) / 60
        print(f"\n  Round {round_num} | Tier {tier} | "
              f"{elapsed_min:.0f}m elapsed, {remaining_min:.0f}m remaining",
              flush=True)

        # Seed phase
        if tier_idx < 7:  # First pass through tiers: seed heavily
            n_seed = params["seed_n"]
            print(f"  Seeding {n_seed} puzzles...", flush=True)
            seed_result = trainer.seed_with_random(n=n_seed, tier=tier)
            print(f"    Solved: {seed_result['solved']}/{seed_result['seeded']}, "
                  f"Q-states: {len(trainer.sarsa.q_table)}", flush=True)

        # GAN round
        n_puzzles = min(params.get("gan_n", 100), 100)
        result = trainer.run(n_puzzles=n_puzzles, verbose=verbose)
        print(f"    GAN: {result['solved']}/{result['generated']} solved, "
              f"Q-states: {len(trainer.sarsa.q_table)}", flush=True)

        stats["rounds"].append({
            "round": round_num,
            "tier": tier,
            "solved": result["solved"],
            "generated": result["generated"],
            "q_states": len(trainer.sarsa.q_table),
            "elapsed_min": round(elapsed_min, 1),
        })
        stats["total_q_states"] = len(trainer.sarsa.q_table)

        if tier_idx < 7:
            stats["tiers_completed"].append(tier)

        tier_idx += 1

        # ── Checkpoint ──
        if time.time() >= t_next_checkpoint:
            _save_checkpoint(trainer.sarsa, sarsa_path, stats, stats_path)
            stats["checkpoints_saved"] += 1
            t_next_checkpoint = time.time() + checkpoint_min * 60
            print(f"  💾 Checkpoint saved ({stats['checkpoints_saved']})", flush=True)

        # Check time
        if time.time() >= t_end:
            print(f"\n  ⏰ Time budget reached ({hours}h)", flush=True)
            break

    # ── Phase 2: Distill → DQN ──
    print(f"\n{'─'*60}")
    print(f"  Phase 2: Distilling SARSA → DQN...")
    print(f"{'─'*60}")

    dqn = LoloDQNLearner(n_actions=6)
    n_params = sum(p.numel() for p in dqn.q_net.parameters())
    print(f"  Network: {n_params:,} parameters")

    distill_result = distill(trainer.sarsa, dqn, epochs=100, batch_size=512)

    print(f"\n  Verifying...")
    verify = verify_distillation(
        trainer.sarsa, dqn,
        n_samples=min(2000, len(trainer.sarsa.q_table))
    )

    # ── Save Everything ──
    dqn.save(weights_path)
    _save_checkpoint(trainer.sarsa, sarsa_path, stats, stats_path)

    # Final stats
    stats["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["total_elapsed_min"] = round((time.time() - t_start) / 60, 1)
    stats["distillation"] = {
        "epochs": distill_result["epochs"],
        "final_loss": distill_result["final_loss"],
        "action_accuracy": verify["accuracy"],
        "q_mse": verify["avg_mse"],
    }
    stats["dqn_weights_kb"] = round(os.path.getsize(weights_path) / 1024, 1)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {stats['total_elapsed_min']:.0f} min")
    print(f"  Q-states: {stats['total_q_states']:,}")
    print(f"  Rounds: {round_num}")
    print(f"  DQN accuracy: {verify['accuracy']:.0%}")
    print(f"  Weights: {weights_path} ({stats['dqn_weights_kb']:.0f} KB)")
    print(f"  Stats: {stats_path}")

    return stats


def _save_checkpoint(sarsa, sarsa_path, stats, stats_path):
    """Save SARSA Q-table and stats."""
    q_data = {}
    for k, v in sarsa.q_table.items():
        q_data[k] = v.copy()
    np.save(sarsa_path, q_data, allow_pickle=True)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def push_weights(output_dir: str):
    """Push weights to GCS if credentials available."""
    try:
        from cloud.gcs_sync import push
        print(f"\n  Pushing weights to GCS...", flush=True)
        push(bucket_name=os.environ.get("GCS_BUCKET", "throng5-weightsb"),
             weights_only=True)
    except Exception as e:
        print(f"\n  ⚠ GCS push failed: {e}")
        print(f"  Weights saved locally in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cloud game trainer")
    parser.add_argument("--game", default="lolo",
                        choices=list(GAME_CONFIGS.keys()),
                        help="Which game to train")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="Training time budget in hours (default: 1)")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N minutes (default: 10)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: brain/games/<game>/)")
    parser.add_argument("--push", action="store_true",
                        help="Push weights to GCS when done")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("brain", "games", args.game)

    print(f"  Game: {args.game}")
    print(f"  Budget: {args.hours}h")
    print(f"  Checkpoint: every {args.checkpoint_interval}m")
    print(f"  Output: {args.output_dir}")
    print(f"  GCS push: {'yes' if args.push else 'no'}")

    if args.game == "lolo":
        stats = train_lolo(
            hours=args.hours,
            checkpoint_min=args.checkpoint_interval,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        print(f"  Game '{args.game}' not yet implemented")
        sys.exit(1)

    if args.push:
        push_weights(args.output_dir)

    print(f"\n  Done! Total cost estimate: ~${args.hours * 0.14:.2f}")


if __name__ == "__main__":
    main()
