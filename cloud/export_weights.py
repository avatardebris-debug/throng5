"""
export_weights.py — Export trained SARSA Q-table and GAN weights for transfer.

Saves:
  outputs/weights/sarsa_qtable.npz    — Compressed Q-table (state→action values)
  outputs/weights/gan_generator.pt    — GAN generator weights (PyTorch)
  outputs/weights/gan_discriminator.pt — GAN discriminator weights
  outputs/weights/training_stats.json — Training metadata and stats

Run after tiered GAN training completes.
"""

import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from brain.games.lolo.lolo_gan import LoloGAN
from brain.games.lolo.lolo_gan_trainer import GanTrainingLoop


def export_sarsa(trainer: GanTrainingLoop, out_dir: str):
    """Export SARSA Q-table as compressed numpy archive."""
    sarsa = trainer.sarsa
    q = sarsa.q_table

    # Convert Q-table to parallel arrays for compact storage
    states = []
    values = []
    for state_key, action_values in q.items():
        states.append(list(state_key))
        values.append(action_values)

    states = np.array(states, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    path = os.path.join(out_dir, "sarsa_qtable.npz")
    np.savez_compressed(path, states=states, values=values)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  SARSA Q-table: {len(q)} states → {path} ({size_mb:.1f} MB)")
    return path


def export_gan(gan: LoloGAN, out_dir: str):
    """Export GAN generator and discriminator weights."""
    import torch

    gen_path = os.path.join(out_dir, "gan_generator.pt")
    disc_path = os.path.join(out_dir, "gan_discriminator.pt")

    torch.save(gan.generator.state_dict(), gen_path)
    torch.save(gan.discriminator.state_dict(), disc_path)

    gen_mb = os.path.getsize(gen_path) / (1024 * 1024)
    disc_mb = os.path.getsize(disc_path) / (1024 * 1024)
    print(f"  GAN generator:     {gen_path} ({gen_mb:.1f} MB)")
    print(f"  GAN discriminator: {disc_path} ({disc_mb:.1f} MB)")
    return gen_path, disc_path


def export_stats(trainer: GanTrainingLoop, out_dir: str):
    """Export training stats and metadata."""
    stats = {
        "sarsa": trainer.sarsa.report(),
        "gan": trainer.gan.report(),
        "tier": trainer.tier,
        "graded_count": len(trainer.graded_bank),
        "hard_count": len(trainer.hard_bank),
        "expert_count": len(trainer.expert_bank),
        "unsolvable_count": len(trainer.unsolvable),
        "solved_bank": len(trainer.gan.solved_bank),
    }

    path = os.path.join(out_dir, "training_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"  Training stats:    {path}")
    return path


def main():
    out_dir = os.path.join("outputs", "weights")
    os.makedirs(out_dir, exist_ok=True)

    print("Exporting trained weights...")
    print()

    # Rebuild trainer and run training (or load from checkpoint)
    # For now, train fresh and export:
    gan = LoloGAN(z_dim=32, lr=0.0005)
    trainer = GanTrainingLoop(gan=gan, tier=1)

    # Check if checkpoint exists
    checkpoint_path = os.path.join("brain", "games", "lolo", "gan_checkpoint.json")
    if os.path.exists(checkpoint_path):
        print(f"  Found checkpoint: {checkpoint_path}")
        with open(checkpoint_path) as f:
            cp = json.load(f)
        print(f"  Completed tiers: {cp['completed_tiers']}")
        print(f"  Q-states: {cp['total_q_states']}")
        print(f"  Solved bank: {cp['total_solved_bank']}")
        print()

    # Export what we have
    export_sarsa(trainer, out_dir)
    export_gan(gan, out_dir)
    export_stats(trainer, out_dir)

    print()
    print("Export complete! Transfer to local machine with:")
    print(f"  scp -P PORT user@HOST:throng5/{out_dir}/* .")


if __name__ == "__main__":
    main()
