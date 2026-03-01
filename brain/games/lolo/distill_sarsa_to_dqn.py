"""
distill_sarsa_to_dqn.py — Transfer SARSA Q-knowledge into DQN via supervised learning.

Reads the SARSA Q-table (435K+ entries keyed by 84-dim compressed state)
and trains the DQN to reproduce those Q-values. This gives the DQN
instant puzzle-solving ability AND generalization to unseen states.

Usage:
  python brain/games/lolo/distill_sarsa_to_dqn.py

Outputs:
  brain/games/lolo/dqn_weights.pt — Pretrained DQN weights
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from brain.games.lolo.lolo_dqn_learner import LoloDQNLearner
from brain.games.lolo.lolo_sarsa_learner import LoloSarsaLearner
from brain.games.lolo.lolo_gan_trainer import GanTrainingLoop
from brain.games.lolo.lolo_gan import LoloGAN


def build_sarsa_dataset(sarsa: LoloSarsaLearner):
    """Extract (state, q_values) pairs from SARSA Q-table."""
    states = []
    q_values = []

    for state_key, action_values in sarsa.q_table.items():
        state_arr = np.array(state_key, dtype=np.float32)
        # Only use entries with meaningful Q-values (not all zeros)
        if np.any(action_values != 0):
            states.append(state_arr)
            q_values.append(action_values.copy())

    states = np.array(states, dtype=np.float32)
    q_values = np.array(q_values, dtype=np.float32)

    print(f"  Dataset: {len(states)} states with non-zero Q-values")
    print(f"  State shape: {states.shape}")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")

    return states, q_values


def distill(
    sarsa: LoloSarsaLearner,
    dqn: LoloDQNLearner,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    """
    Train DQN to reproduce SARSA's Q-values via supervised learning.

    Returns training stats dict.
    """
    states, q_values = build_sarsa_dataset(sarsa)

    if len(states) == 0:
        print("  ⚠ SARSA Q-table is empty — nothing to distill")
        return {"epochs": 0, "final_loss": float("inf")}

    # Convert to tensors
    device = dqn.device
    s_tensor = torch.tensor(states, device=device)
    q_tensor = torch.tensor(q_values, device=device)

    # Use DQN's own optimizer (or create a new one for distillation)
    optimizer = optim.Adam(dqn.q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = len(states)
    steps_per_epoch = max(1, n_samples // batch_size)

    best_loss = float("inf")
    losses = []

    t0 = time.time()

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            s_batch = s_tensor[idx]
            q_batch = q_tensor[idx]

            predicted = dqn.q_net(s_batch)
            loss = loss_fn(predicted, q_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.q_net.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        losses.append(avg_loss)
        best_loss = min(best_loss, avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f} "
                  f"best={best_loss:.6f} ({elapsed:.1f}s)", flush=True)

    # Sync target network
    dqn.target_net.load_state_dict(dqn.q_net.state_dict())

    elapsed = time.time() - t0
    print(f"\n  Distillation done: {elapsed:.1f}s, final loss={losses[-1]:.6f}")

    return {
        "epochs": epochs,
        "samples": n_samples,
        "final_loss": losses[-1],
        "best_loss": best_loss,
        "elapsed": round(elapsed, 1),
    }


def verify_distillation(sarsa: LoloSarsaLearner, dqn: LoloDQNLearner,
                         n_samples: int = 1000) -> dict:
    """Verify DQN matches SARSA's action choices."""
    states, q_values = build_sarsa_dataset(sarsa)

    if len(states) < n_samples:
        n_samples = len(states)

    # Random sample
    idx = np.random.choice(len(states), n_samples, replace=False)

    match_count = 0
    total_q_error = 0.0

    for i in idx:
        state = states[i]
        sarsa_q = q_values[i]
        sarsa_action = int(np.argmax(sarsa_q))

        dqn_q = dqn.get_q_values(state)
        dqn_action = int(np.argmax(dqn_q))

        if sarsa_action == dqn_action:
            match_count += 1
        total_q_error += float(np.mean((dqn_q - sarsa_q) ** 2))

    accuracy = match_count / n_samples
    avg_mse = total_q_error / n_samples

    print(f"\n  Verification ({n_samples} states):")
    print(f"    Action match: {match_count}/{n_samples} ({accuracy:.0%})")
    print(f"    Avg Q-MSE: {avg_mse:.6f}")

    return {
        "accuracy": accuracy,
        "avg_mse": avg_mse,
        "n_samples": n_samples,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distill SARSA → DQN")
    parser.add_argument("--skip-sarsa", action="store_true",
                        help="Load saved Q-table instead of retraining SARSA")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Distillation epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SARSA → DQN Distillation")
    print("=" * 60)

    sarsa_path = os.path.join("brain", "games", "lolo", "sarsa_qtable.npy")

    # ── Step 1: Get SARSA Q-table ──
    if args.skip_sarsa and os.path.exists(sarsa_path):
        print(f"\n  Step 1: Loading saved Q-table from {sarsa_path}...")
        q_data = np.load(sarsa_path, allow_pickle=True).item()

        # Build a minimal SARSA wrapper with the loaded Q-table
        sarsa = LoloSarsaLearner(n_actions=6)
        sarsa.q_table = q_data
        print(f"  Loaded: {len(sarsa.q_table)} Q-states")
    else:
        print("\n  Step 1: Building SARSA Q-table via GAN training...")

        gan = LoloGAN(z_dim=32, lr=0.0005)
        trainer = GanTrainingLoop(gan=gan, tier=1, use_dqn=False)

        from brain.games.lolo.run_tiered_gan import TIER_PARAMS

        for tier in range(1, 8):
            params = TIER_PARAMS[tier]
            trainer.tier = tier

            print(f"\n  Tier {tier}: Seeding {params['seed_n']} puzzles...")
            seed_result = trainer.seed_with_random(n=params["seed_n"], tier=tier)
            solved = seed_result["solved"]
            total = seed_result["seeded"]
            print(f"    Solved: {solved}/{total} ({solved/max(total,1):.0%}), "
                  f"Q-states: {len(trainer.sarsa.q_table)}")

            result = trainer.run(n_puzzles=50, verbose=False)
            print(f"    GAN round: {result['solved']}/{result['generated']} solved")

        sarsa = trainer.sarsa
        print(f"\n  SARSA built: {len(sarsa.q_table)} Q-states")

        # Save Q-table for next time
        q_data = {}
        for k, v in sarsa.q_table.items():
            q_data[k] = v.copy()
        np.save(sarsa_path, q_data, allow_pickle=True)
        sarsa_kb = os.path.getsize(sarsa_path) / 1024
        print(f"  ✅ SARSA Q-table saved: {sarsa_path} ({sarsa_kb:.0f} KB)")

    # ── Step 2: Create DQN and distill ──
    print(f"\n  Step 2: Distilling SARSA → DQN ({args.epochs} epochs, batch={args.batch_size})...")
    dqn = LoloDQNLearner(n_actions=6)
    n_params = sum(p.numel() for p in dqn.q_net.parameters())
    print(f"  Network: {n_params:,} parameters")

    result = distill(sarsa, dqn, epochs=args.epochs, batch_size=args.batch_size)

    # ── Step 3: Verify ──
    print("\n  Step 3: Verifying distillation...")
    verify = verify_distillation(sarsa, dqn, n_samples=min(2000, len(sarsa.q_table)))

    # ── Step 4: Save weights ──
    weights_path = os.path.join("brain", "games", "lolo", "dqn_weights.pt")
    dqn.save(weights_path)
    size_kb = os.path.getsize(weights_path) / 1024
    print(f"\n  ✅ DQN weights saved: {weights_path} ({size_kb:.0f} KB)")
    print(f"     Action accuracy: {verify['accuracy']:.0%}")
    print(f"     Q-value MSE: {verify['avg_mse']:.6f}")


if __name__ == "__main__":
    main()

