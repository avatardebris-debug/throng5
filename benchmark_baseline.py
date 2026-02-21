"""
benchmark_baseline.py
=====================
Run N episodes of each game with a freshly-initialized PortableNNAgent
(pure RL, no human data).  Results are saved to benchmark_results/ as
JSON so they can be compared against the Option-A (human-seeded) run.

Usage
-----
    # Default: 200 episodes on games you've played
    python benchmark_baseline.py

    # Custom game list and episode count
    python benchmark_baseline.py --games ALE/Breakout-v5 ALE/SpaceInvaders-v5 --episodes 50

    # Skip already-completed games (resume)
    python benchmark_baseline.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from throng4.learning.portable_agent import PortableNNAgent, AgentConfig

# ── output dir ────────────────────────────────────────────────────────
RESULTS_DIR = _ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── games the user has played (auto-detected from DB if available) ────
DEFAULT_GAMES = [
    "ALE/Breakout-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/Frogger-v5",
    "ALE/MarioBros-v5",
]


# ─────────────────────────────────────────────────────────────────────
# One episode: agent selects actions, env steps, return accumulated
# ─────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    agent: PortableNNAgent,
    n_actions: int,
    max_steps: int = 10_000,
    train: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run one episode. Returns dict with reward, steps, and training flag.
    Agent selects greedily (epsilon-greedy) among all valid actions.
    Feature vector: [RAM/255 (128-dim) | one-hot action (n_actions-dim)]
    """
    obs, _ = env.reset(seed=seed)
    ram = np.array(obs, dtype=np.float32) / 255.0

    total_reward = 0.0
    steps = 0
    episode_buf: list[tuple] = []  # (feat, reward, done)

    done = False
    while not done and steps < max_steps:
        # Build feature for each action and pick best
        best_action = 0
        if agent.rng.rand() < agent.epsilon:
            best_action = int(agent.rng.randint(n_actions))
        else:
            best_val = -float("inf")
            for a in range(n_actions):
                ah = np.zeros(n_actions, dtype=np.float32)
                ah[a] = 1.0
                feat = np.concatenate([ram, ah])
                val = agent.forward(feat)
                if val > best_val:
                    best_val = val
                    best_action = a

        # Step
        obs, reward, terminated, truncated, _ = env.step(best_action)
        done = terminated or truncated
        next_ram = np.array(obs, dtype=np.float32) / 255.0

        # Build feature for best next action (for replay)
        next_feats = []
        for a in range(n_actions):
            ah = np.zeros(n_actions, dtype=np.float32)
            ah[a] = 1.0
            nf = np.concatenate([next_ram, ah])
            # Guard against NaN/Inf before pushing
            if np.isfinite(nf).all():
                next_feats.append(nf)

        # Current feature
        cur_ah = np.zeros(n_actions, dtype=np.float32)
        cur_ah[best_action] = 1.0
        cur_feat = np.concatenate([ram, cur_ah])

        # Push to replay buffer
        agent.replay_buffer.push(cur_feat, float(reward), next_feats, done)
        episode_buf.append((cur_feat, float(reward)))

        total_reward += reward
        steps += 1
        ram = next_ram

        # Train periodically
        if (train
                and steps % agent.config.train_freq == 0
                and len(agent.replay_buffer) >= agent.config.batch_size):
            agent._train_batch()

    # Epsilon decay at episode end
    agent.epsilon = max(agent.config.epsilon_min,
                        agent.epsilon * agent.config.epsilon_decay)
    agent.episode_count += 1

    return {
        "reward":  float(total_reward),
        "steps":   steps,
        "epsilon": float(agent.epsilon),
    }


# ─────────────────────────────────────────────────────────────────────
# Benchmark one game
# ─────────────────────────────────────────────────────────────────────

def benchmark_game(
    game_id: str,
    n_episodes: int,
    seed_base: int = 42,
    resume: bool = False,
) -> dict[str, Any]:
    """Run n_episodes of game_id, return summary + per-episode results."""
    slug = game_id.replace("/", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"baseline_{slug}.json"

    if resume and out_path.exists():
        print(f"  [skip] {game_id} — already done ({out_path.name})")
        return json.loads(out_path.read_text())

    print(f"\n{'='*60}")
    print(f"  BASELINE: {game_id}  ({n_episodes} episodes)")
    print(f"{'='*60}")

    env = gym.make(game_id, obs_type="ram", render_mode=None)
    n_actions = env.action_space.n
    n_features = 128 + n_actions   # RAM + one-hot action

    cfg = AgentConfig(
        n_hidden=256,
        n_hidden2=128,
        epsilon=0.20,
        epsilon_decay=0.995,
        epsilon_min=0.02,
        gamma=0.95,
        learning_rate=0.005,
        replay_buffer_size=50_000,
        batch_size=64,
        train_freq=4,
    )
    agent = PortableNNAgent(n_features, config=cfg, seed=seed_base)

    episodes: list[dict] = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_seed = seed_base + ep
        result = run_episode(env, agent, n_actions, seed=ep_seed)
        result["episode"] = ep
        episodes.append(result)

        if (ep + 1) % 10 == 0 or ep == 0:
            recent = episodes[-20:]
            avg_r = np.mean([e["reward"] for e in recent])
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(f"  ep {ep+1:4d}/{n_episodes}  "
                  f"avg_reward(last20)={avg_r:7.2f}  "
                  f"eps={agent.epsilon:.3f}  "
                  f"eta={eta/60:.1f}min")

    env.close()
    elapsed = time.time() - t0

    # Summary stats
    rewards = [e["reward"] for e in episodes]
    summary = {
        "game":        game_id,
        "arm":         "baseline",
        "n_episodes":  n_episodes,
        "elapsed_s":   round(elapsed, 1),
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "max_reward":  float(np.max(rewards)),
        "min_reward":  float(np.min(rewards)),
        "p25_reward":  float(np.percentile(rewards, 25)),
        "p75_reward":  float(np.percentile(rewards, 75)),
        "episodes":    episodes,
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Done. mean={summary['mean_reward']:.2f} "
          f"max={summary['max_reward']:.2f}  "
          f"saved → {out_path.name}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Summary table after all games
# ─────────────────────────────────────────────────────────────────────

def print_summary(summaries: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Game':<35} {'Mean':>8} {'Std':>7} {'Max':>8} {'N':>5}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*8} {'-'*5}")
    for s in summaries:
        name = s['game'].replace('ALE/', '')
        print(f"  {name:<35} {s['mean_reward']:>8.2f} "
              f"{s['std_reward']:>7.2f} {s['max_reward']:>8.2f} "
              f"{s['n_episodes']:>5}")

    # Save combined summary
    combined = RESULTS_DIR / "baseline_summary.json"
    combined.write_text(json.dumps(summaries, indent=2))
    print(f"\n  Full results saved → {combined}")


# ─────────────────────────────────────────────────────────────────────
# Detect games from replay DB
# ─────────────────────────────────────────────────────────────────────

def detect_played_games() -> list[str]:
    """Read replay_db.sqlite to find which games the user has played."""
    db_path = _ROOT / "experiments" / "replay_db.sqlite"
    if not db_path.exists():
        return []
    try:
        import sqlite3
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            "SELECT DISTINCT env_name FROM sessions WHERE env_name IS NOT NULL"
        ).fetchall()
        con.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Baseline benchmark: pure RL with no human data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", nargs="+", default=None,
                   help="Game IDs to benchmark. Auto-detected from replay DB if omitted.")
    p.add_argument("--episodes", type=int, default=200,
                   help="Episodes per game")
    p.add_argument("--seed", type=int, default=42,
                   help="Base RNG seed")
    p.add_argument("--resume", action="store_true",
                   help="Skip games already benchmarked")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    # Resolve game list
    games = args.games
    if not games:
        games = detect_played_games()
        if games:
            print(f"[auto] Found games in replay DB: {games}")
        else:
            print(f"[auto] No replay DB found, using defaults: {DEFAULT_GAMES}")
            games = DEFAULT_GAMES

    print(f"\nBenchmarking {len(games)} game(s) x {args.episodes} episodes each")
    print(f"Results → {RESULTS_DIR}/\n")

    summaries = []
    for game in games:
        try:
            s = benchmark_game(
                game,
                n_episodes=args.episodes,
                seed_base=args.seed,
                resume=args.resume,
            )
            summaries.append(s)
        except Exception as e:
            print(f"  ERROR on {game}: {e}")

    if summaries:
        print_summary(summaries)
