"""
load_weights.py — Load cloud-trained weights into local brain.

Loads:
  sarsa_qtable.npz     → LoloSarsaLearner Q-table
  gan_generator.pt     → LoloGAN generator
  gan_discriminator.pt → LoloGAN discriminator

Usage:
  python cloud/load_weights.py outputs/weights/
"""

import json
import os
import sys
from typing import Dict

sys.path.insert(0, ".")

import numpy as np


def load_sarsa_qtable(path: str) -> Dict[tuple, np.ndarray]:
    """Load SARSA Q-table from compressed numpy archive."""
    data = np.load(path)
    states = data["states"]
    values = data["values"]

    q_table = {}
    for i in range(len(states)):
        key = tuple(states[i].astype(np.float32))
        q_table[key] = values[i].copy()

    print(f"  Loaded {len(q_table)} Q-states from {path}")
    return q_table


def load_gan_weights(gen_path: str, disc_path: str):
    """Load GAN weights into a new LoloGAN instance."""
    import torch
    from brain.games.lolo.lolo_gan import LoloGAN

    gan = LoloGAN(z_dim=32, lr=0.0005)
    gan.generator.load_state_dict(torch.load(gen_path, weights_only=True))
    gan.discriminator.load_state_dict(torch.load(disc_path, weights_only=True))
    print(f"  Loaded GAN generator from {gen_path}")
    print(f"  Loaded GAN discriminator from {disc_path}")
    return gan


def load_into_sarsa(weight_dir: str):
    """Load weights into a fresh SARSA learner."""
    from brain.games.lolo.lolo_sarsa_learner import LoloSarsaLearner

    sarsa = LoloSarsaLearner()
    qtable_path = os.path.join(weight_dir, "sarsa_qtable.npz")
    if os.path.exists(qtable_path):
        sarsa.q_table = load_sarsa_qtable(qtable_path)
        sarsa.epsilon = 0.05  # Low exploration since we have pretrained knowledge
        print(f"  SARSA ready: {len(sarsa.q_table)} states, epsilon=0.05")
    return sarsa


def main():
    if len(sys.argv) < 2:
        print("Usage: python cloud/load_weights.py <weight_dir>")
        print("  e.g.: python cloud/load_weights.py outputs/weights/")
        sys.exit(1)

    weight_dir = sys.argv[1]
    print(f"Loading weights from: {weight_dir}")
    print()

    # Load SARSA
    sarsa = load_into_sarsa(weight_dir)

    # Load GAN
    gen_path = os.path.join(weight_dir, "gan_generator.pt")
    disc_path = os.path.join(weight_dir, "gan_discriminator.pt")
    if os.path.exists(gen_path):
        gan = load_gan_weights(gen_path, disc_path)
    else:
        print("  ⚠ GAN weights not found, skipping")

    # Load stats
    stats_path = os.path.join(weight_dir, "training_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n  Training stats:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    print("\n  ✓ All weights loaded successfully!")
    print("  SARSA Q-table is ready for basal ganglia integration")


if __name__ == "__main__":
    main()
