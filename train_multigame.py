"""
train_multigame.py — Multi-game curriculum training with transfer learning.

Trains a single WholeBrain instance across multiple games in sequence,
transferring learned representations between games.

Default curriculum: CartPole(200ep) → Pong(500ep) → Breakout(500ep) → Montezuma(1000ep)

Usage:
    python train_multigame.py --curriculum cartpole,pong,breakout
    python train_multigame.py --curriculum cartpole:200,pong:500 --transfer-cnn
    python train_multigame.py --dream-every 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from brain.orchestrator import WholeBrain


# ── Default Curriculum ────────────────────────────────────────────────

DEFAULT_CURRICULUM = [
    ("cartpole", 200),
    ("pong", 500),
    ("breakout", 500),
    ("montezuma", 1000),
]

# Game configs: n_actions and adapter factory
GAME_CONFIGS = {
    "cartpole":   {"n_actions": 2,  "n_features": 84, "use_cnn": False},
    "pong":       {"n_actions": 6,  "n_features": 84, "use_cnn": True},
    "breakout":   {"n_actions": 4,  "n_features": 84, "use_cnn": True},
    "montezuma": {"n_actions": 18, "n_features": 84, "use_cnn": True},
    "tetris":     {"n_actions": 6,  "n_features": 84, "use_cnn": False},
}


def parse_curriculum(spec: str) -> List[Tuple[str, int]]:
    """Parse curriculum string like 'cartpole:200,pong:500'."""
    curriculum = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        game = parts[0].strip().lower()
        episodes = int(parts[1]) if len(parts) > 1 else 500
        curriculum.append((game, episodes))
    return curriculum


def create_env(game_name: str):
    """Create environment for a game. Returns (env, adapter)."""
    game_name = game_name.lower()

    if game_name == "cartpole":
        try:
            import gymnasium as gym
            from brain.environments.gym_envs import GymAdapter
            env = gym.make("CartPole-v1")
            adapter = GymAdapter(env)
            return env, adapter
        except ImportError:
            print(f"  [WARN] gymnasium not available, using dummy env for {game_name}")
            return _create_dummy_env(game_name)

    elif game_name in ("pong", "breakout", "montezuma"):
        try:
            from brain.environments.atari_adapter import AtariAdapter
            rom_map = {
                "pong": "Pong",
                "breakout": "Breakout",
                "montezuma": "MontezumaRevenge",
            }
            adapter = AtariAdapter(rom_name=rom_map[game_name])
            return adapter, adapter
        except ImportError:
            print(f"  [WARN] Atari adapter not available, using dummy env for {game_name}")
            return _create_dummy_env(game_name)

    elif game_name == "tetris":
        try:
            from brain.environments.tetris_adapter import TetrisAdapter
            adapter = TetrisAdapter()
            return adapter, adapter
        except ImportError:
            print(f"  [WARN] Tetris adapter not available, using dummy env for {game_name}")
            return _create_dummy_env(game_name)

    else:
        print(f"  [WARN] Unknown game '{game_name}', using dummy env")
        return _create_dummy_env(game_name)


def _create_dummy_env(game_name: str):
    """Create a dummy environment for testing curriculum logic."""
    config = GAME_CONFIGS.get(game_name, {"n_actions": 4, "n_features": 84})

    class DummyEnv:
        def __init__(self):
            self.n_actions = config["n_actions"]
            self.n_features = config["n_features"]
            self._step_count = 0

        def reset(self):
            self._step_count = 0
            return np.random.randn(self.n_features).astype(np.float32)

        def step(self, action):
            self._step_count += 1
            obs = np.random.randn(self.n_features).astype(np.float32)
            reward = np.random.randn() * 0.5
            done = self._step_count >= 100
            return obs, reward, done, {}

    env = DummyEnv()
    return env, None


def train_game(
    brain: WholeBrain,
    game_name: str,
    n_episodes: int,
    env,
    adapter,
    dream_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train brain on a single game for n_episodes.

    Returns metrics dict.
    """
    if adapter is not None:
        brain.set_adapter(adapter)

    config = GAME_CONFIGS.get(game_name, {})
    episode_rewards = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        action = 0
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            result = brain.step(obs, prev_action=action, reward=0.0, done=False)
            action = result["action"] % config.get("n_actions", 4)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if ep_steps > 10000:
                done = True

        # Final step with done=True
        brain.step(obs, prev_action=action, reward=reward, done=True)
        episode_rewards.append(ep_reward)

        if verbose and (ep + 1) % 50 == 0:
            recent = episode_rewards[-50:]
            avg = sum(recent) / len(recent)
            print(f"  [{game_name}] Ep {ep+1}/{n_episodes}: "
                  f"avg_reward={avg:.2f}, last={ep_reward:.2f}")

        # Periodic dreaming
        if dream_every > 0 and (ep + 1) % dream_every == 0:
            try:
                from brain.overnight.dream_loop import DreamLoop
                dreamer = DreamLoop(brain)
                dream_report = dreamer.run(
                    n_replay_cycles=10,
                    n_dream_steps=10,
                    max_time_seconds=60.0,
                )
                if verbose:
                    print(f"  [{game_name}] Dream cycle: "
                          f"batches={dream_report['phase_a_replay'].get('batches_processed', 0)}")
            except Exception as e:
                if verbose:
                    print(f"  [{game_name}] Dream failed: {e}")

    elapsed = time.perf_counter() - t0

    return {
        "game": game_name,
        "episodes": n_episodes,
        "total_steps": brain._step_count,
        "avg_reward": round(sum(episode_rewards) / max(len(episode_rewards), 1), 4),
        "final_50_avg": round(sum(episode_rewards[-50:]) / max(len(episode_rewards[-50:]), 1), 4),
        "elapsed_seconds": round(elapsed, 2),
        "steps_per_second": round(brain._step_count / max(elapsed, 0.01), 1),
    }


def save_checkpoint(brain, game_name: str, checkpoint_dir: str) -> str:
    """Save brain checkpoint between games."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"brain_{game_name}.pt")

    try:
        from brain.scaling.brain_state import BrainState
        BrainState.save(brain, path)
        return path
    except ImportError:
        # Fallback: just save a marker
        with open(path + ".json", "w") as f:
            json.dump({"game": game_name, "steps": brain._step_count}, f)
        return path + ".json"


def main():
    parser = argparse.ArgumentParser(description="Multi-game curriculum training")
    parser.add_argument(
        "--curriculum", type=str,
        default="cartpole:200,pong:500,breakout:500",
        help="Comma-separated game:episodes pairs",
    )
    parser.add_argument("--transfer-cnn", action="store_true", help="Transfer CNN between games")
    parser.add_argument("--freeze-layers", type=int, default=2, help="CNN layers to freeze on transfer")
    parser.add_argument("--dream-every", type=int, default=0, help="Dream every N episodes (0=disabled)")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/multigame", help="Checkpoint directory")
    parser.add_argument("--use-torch", action="store_true", default=True, help="Use PyTorch DQN")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    curriculum = parse_curriculum(args.curriculum)
    print(f"Multi-Game Curriculum Training")
    print(f"  Games: {' → '.join(f'{g}({n}ep)' for g, n in curriculum)}")
    print(f"  Transfer CNN: {args.transfer_cnn}")
    print(f"  Dream every: {args.dream_every} episodes")
    print()

    # Use first game's config to initialize
    first_game = curriculum[0][0]
    first_config = GAME_CONFIGS.get(first_game, {"n_actions": 4, "n_features": 84})

    brain = WholeBrain(
        n_features=first_config["n_features"],
        n_actions=first_config["n_actions"],
        use_torch=args.use_torch,
        use_cnn=first_config.get("use_cnn", False),
    )

    all_results = []

    for i, (game_name, n_episodes) in enumerate(curriculum):
        print(f"{'='*60}")
        print(f"Game {i+1}/{len(curriculum)}: {game_name.upper()} ({n_episodes} episodes)")
        print(f"{'='*60}")

        # Create environment
        env, adapter = create_env(game_name)

        # Transfer learning from previous game
        if i > 0 and args.transfer_cnn:
            try:
                from brain.learning.transfer import transfer_cnn
                result = transfer_cnn(brain, brain, freeze_layers=args.freeze_layers)
                print(f"  CNN transfer: {result.get('n_transferred', 0)} params transferred, "
                      f"{result.get('n_frozen', 0)} frozen")
            except Exception as e:
                print(f"  CNN transfer skipped: {e}")

        # Train
        result = train_game(
            brain, game_name, n_episodes,
            env, adapter,
            dream_every=args.dream_every,
            verbose=args.verbose,
        )
        all_results.append(result)

        print(f"  Result: avg_reward={result['avg_reward']}, "
              f"final_50_avg={result['final_50_avg']}, "
              f"elapsed={result['elapsed_seconds']}s")

        # Save checkpoint
        ckpt = save_checkpoint(brain, game_name, args.checkpoint_dir)
        print(f"  Checkpoint: {ckpt}")
        print()

    # Final summary
    print(f"\n{'='*60}")
    print("CURRICULUM COMPLETE")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['game']:12s}: avg={r['avg_reward']:8.2f}  final={r['final_50_avg']:8.2f}  "
              f"time={r['elapsed_seconds']:6.1f}s  rate={r['steps_per_second']:.0f} steps/s")

    # Save full report
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    report_path = os.path.join(args.checkpoint_dir, "curriculum_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReport saved: {report_path}")

    brain.close()


if __name__ == "__main__":
    main()
