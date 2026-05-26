#!/usr/bin/env python3
"""
PufferLib-style fast pretrain → Throng5 Striatum checkpoint bridge.

Trains RainbowDQN on 84-dim abstract Atari features (same as WholeBrain) using
PufferLib vectorized envs when available, exports a checkpoint, and optionally
loads it into WholeBrain.

Examples:
  # Pretrain Breakout (fast vector rollouts)
  python tools/puffer_pretrain.py train --game ALE/Breakout-v5 --steps 200000

  # Load weights into WholeBrain and run a short eval
  python tools/puffer_pretrain.py load --checkpoint checkpoints/puffer_pretrain.pt --steps 500

  # Pretrain then load in one command
  python tools/puffer_pretrain.py train-and-load --game ALE/MontezumaRevenge-v5 --steps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()

import numpy as np

from brain.bridge import TrainConfig, load_into_brain
from brain.bridge.checkpoint import checkpoint_metadata
from brain.bridge.puffer_dqn_trainer import PufferDQNTrainer
from brain.orchestrator import WholeBrain


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        game_id=args.game,
        total_steps=args.steps,
        num_envs=args.num_envs,
        num_workers=args.workers,
        batch_size=args.batch_size,
        train_interval=args.train_interval,
        log_interval=args.log_interval,
        export_path=args.export,
        device=args.device,
        use_puffer_vector=not args.no_puffer,
    )
    stats = PufferDQNTrainer(cfg).train()
    print("\n=== Pretrain complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _cmd_load(args: argparse.Namespace) -> None:
    meta = checkpoint_metadata(args.checkpoint)
    n_actions = args.n_actions or meta.get("n_actions") or 18
    print(f"Checkpoint: game={meta.get('game_id')} n_actions={n_actions} updates={meta.get('total_updates')}")

    brain = WholeBrain(
        n_features=84,
        n_actions=n_actions,
        session_name="puffer_pretrain_load",
        enable_logging=args.log,
        use_torch=True,
    )
    info = load_into_brain(brain, args.checkpoint)
    print("Loaded checkpoint:", info)

    if args.steps <= 0:
        brain.close()
        return

    # Smoke-only: random 84-d vectors, not real Atari obs (use FeatureAtariEnv for that).
    rng = np.random.RandomState(args.seed)
    prev_action = 0
    for i in range(args.steps):
        obs = rng.randn(84).astype(np.float32)
        result = brain.step(
            obs,
            prev_action=prev_action,
            reward=0.0,
            done=(i > 0 and i % 200 == 0),
        )
        prev_action = int(result["action"])
        if (i + 1) % 100 == 0:
            print(f"  step {i+1} action={prev_action} eps={result.get('epsilon', 0):.3f}")

    brain.close()
    print(f"Eval finished ({args.steps} steps)")


def _cmd_train_and_load(args: argparse.Namespace) -> None:
    _cmd_train(args)
    args.checkpoint = args.export
    args.steps = args.eval_steps
    _cmd_load(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Puffer pretrain → Throng5 bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Vectorized DQN pretrain + export .pt")
    p_train.add_argument("--game", default="ALE/Breakout-v5")
    p_train.add_argument("--steps", type=int, default=500_000)
    p_train.add_argument("--num-envs", type=int, default=32)
    p_train.add_argument("--workers", type=int, default=1, help="Puffer vector workers (1=Serial)")
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--train-interval", type=int, default=4)
    p_train.add_argument("--log-interval", type=int, default=10_000)
    p_train.add_argument("--export", default="checkpoints/puffer_pretrain.pt")
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--no-puffer", action="store_true", help="Force single-env serial loop")
    p_train.set_defaults(func=_cmd_train)

    p_load = sub.add_parser("load", help="Load checkpoint into WholeBrain")
    p_load.add_argument("--checkpoint", required=True)
    p_load.add_argument("--steps", type=int, default=0, help="Optional random-feature smoke steps")
    p_load.add_argument("--n-actions", type=int, default=18)
    p_load.add_argument("--seed", type=int, default=0)
    p_load.add_argument("--log", action="store_true")
    p_load.set_defaults(func=_cmd_load)

    p_both = sub.add_parser("train-and-load", help="Train, export, load, optional eval")
    p_both.add_argument("--game", default="ALE/Breakout-v5")
    p_both.add_argument("--steps", type=int, default=200_000)
    p_both.add_argument("--num-envs", type=int, default=32)
    p_both.add_argument("--workers", type=int, default=1)
    p_both.add_argument("--batch-size", type=int, default=32)
    p_both.add_argument("--train-interval", type=int, default=4)
    p_both.add_argument("--log-interval", type=int, default=10_000)
    p_both.add_argument("--export", default="checkpoints/puffer_pretrain.pt")
    p_both.add_argument("--device", default=None)
    p_both.add_argument("--no-puffer", action="store_true")
    p_both.add_argument("--eval-steps", type=int, default=300)
    p_both.add_argument("--n-actions", type=int, default=18)
    p_both.add_argument("--seed", type=int, default=0)
    p_both.add_argument("--log", action="store_true")
    p_both.set_defaults(func=_cmd_train_and_load)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
