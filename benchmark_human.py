"""
benchmark_human.py
==================
Option A benchmark: same RL training as baseline but the replay buffer
is pre-seeded with human play transitions from replay_db.sqlite.
An optional imitation warm-up phase runs gradient steps on the human
actions before RL episodes begin.

Results are saved to benchmark_results/human_<game>.json.
Run compare_benchmarks.py afterwards to see the side-by-side table.

Usage
-----
    python benchmark_human.py                    # all games in DB, 200 eps
    python benchmark_human.py --episodes 50      # quick test
    python benchmark_human.py --warmup 0         # skip imitation warm-up
    python benchmark_human.py --resume           # skip already-done games
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
from throng4.learning.prioritized_replay import PrioritizedReplayBuffer

DB_PATH     = str(_ROOT / "experiments" / "replay_db.sqlite")
RESULTS_DIR = _ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Reuse run_episode logic from baseline (copied to avoid import cycle)
# ─────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    agent: PortableNNAgent,
    buf: PrioritizedReplayBuffer,
    n_actions: int,
    max_steps: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    ram = np.array(obs, dtype=np.float32) / 255.0
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        if agent.rng.rand() < agent.epsilon:
            best_action = int(agent.rng.randint(n_actions))
        else:
            best_val = -float("inf")
            best_action = 0
            for a in range(n_actions):
                ah = np.zeros(n_actions, dtype=np.float32)
                ah[a] = 1.0
                val = agent.forward(np.concatenate([ram, ah]))
                if val > best_val:
                    best_val = val
                    best_action = a

        obs, reward, terminated, truncated, _ = env.step(best_action)
        done = terminated or truncated
        next_ram = np.array(obs, dtype=np.float32) / 255.0

        # Build next feature list
        next_feats = []
        for a in range(n_actions):
            ah = np.zeros(n_actions, dtype=np.float32)
            ah[a] = 1.0
            nf = np.concatenate([next_ram, ah])
            if np.isfinite(nf).all():
                next_feats.append(nf)

        cur_ah = np.zeros(n_actions, dtype=np.float32)
        cur_ah[best_action] = 1.0
        cur_feat = np.concatenate([ram, cur_ah])

        buf.push(cur_feat, float(reward), next_feats, done)

        total_reward += reward
        steps += 1
        ram = next_ram

        # Train from the prioritized buffer
        if steps % agent.config.train_freq == 0 and len(buf) >= agent.config.batch_size:
            batch = buf.sample(agent.config.batch_size)
            _train_on_batch(agent, batch)

    agent.epsilon = max(agent.config.epsilon_min,
                        agent.epsilon * agent.config.epsilon_decay)
    agent.episode_count += 1
    return {"reward": float(total_reward), "steps": steps,
            "epsilon": float(agent.epsilon)}


def _train_on_batch(agent: PortableNNAgent, batch) -> None:
    """One gradient update on a pre-drawn batch (same logic as _train_batch)."""
    agent._training = True
    try:
        for x, r, next_x_list, done in batch:
            if done or not next_x_list:
                target = r if done else r - 10.0
            else:
                max_q = max(agent.forward_target(nx) for nx in next_x_list)
                max_q = np.clip(max_q, -500.0, 500.0)
                target = np.clip(r + agent.config.gamma * max_q, -500.0, 500.0)

            x_noisy = agent._apply_ext_noise(x)
            pred = agent.forward(x_noisy)
            error = pred - float(target)
            ce = np.clip(error, -5, 5)

            agent.W3 -= agent.config.learning_rate * ce * agent._last_h2
            agent.b3 -= agent.config.learning_rate * ce

            dh2 = ce * agent.W3[0] * (agent._last_h2 > 0)
            agent.W2 -= agent.config.learning_rate * np.outer(dh2, agent._last_h1)
            agent.b2 -= agent.config.learning_rate * dh2

            dh1 = (agent.W2.T @ dh2) * (agent._last_h1 > 0)
            agent.W1 -= agent.config.learning_rate * np.outer(dh1, agent._last_x)
            agent.b1 -= agent.config.learning_rate * dh1
    finally:
        agent._training = False
    agent.total_updates += 1


# ─────────────────────────────────────────────────────────────────────
# Imitation warm-up: gradient steps on (state, human_action) pairs
# ─────────────────────────────────────────────────────────────────────

def imitation_warmup(
    agent: PortableNNAgent,
    buf: PrioritizedReplayBuffer,
    n_steps: int,
    n_actions: int,
) -> int:
    """
    Run n_steps of cross-entropy imitation gradient on disagree transitions.
    Only updates the imitation head (Wi1/Wi2) if enabled; otherwise does
    a lightweight behaviour-clone update on the main network output layer
    using the human action as a pseudo-reward target.
    """
    if n_steps <= 0 or len(buf) == 0:
        return 0

    # Collect all disagree transitions from the seeded buffer
    disagree_entries = [e for e in buf._buffer if e.human_agent_disagree]
    if not disagree_entries:
        # Fall back to all entries with a human action
        disagree_entries = [e for e in buf._buffer
                            if e.human_action is not None]
    if not disagree_entries:
        return 0

    rng = agent.rng
    done_steps = 0

    if agent.config.use_imitation_head:
        for _ in range(n_steps):
            entry = disagree_entries[rng.randint(len(disagree_entries))]
            if entry.human_action is not None and 0 <= entry.human_action < n_actions:
                agent._train_imitation_step(entry.x, entry.human_action)
                done_steps += 1
    else:
        # Behaviour-clone lite: nudge the value estimate for the human action
        # higher than average on this state, using a pseudo-reward of +1.0.
        lr = agent.config.learning_rate * 0.5   # gentler than RL lr
        for _ in range(n_steps):
            entry = disagree_entries[rng.randint(len(disagree_entries))]
            ha = entry.human_action
            if ha is None or not (0 <= ha < n_actions):
                continue
            # Build feature for the human's chosen action
            ram = entry.x[:128]
            ah = np.zeros(n_actions, dtype=np.float32)
            ah[ha] = 1.0
            feat = np.concatenate([ram, ah])
            # Target: current estimate + 1.0 (encourage this action)
            pred = agent.forward(feat)
            target = np.clip(pred + 1.0, -500.0, 500.0)
            error = pred - target
            ce = np.clip(error, -5, 5)
            agent.W3 -= lr * ce * agent._last_h2
            agent.b3 -= lr * ce
            done_steps += 1

    return done_steps


# ─────────────────────────────────────────────────────────────────────
# Benchmark one game with human-seeded buffer
# ─────────────────────────────────────────────────────────────────────

def benchmark_game(
    game_id: str,
    n_episodes: int,
    n_warmup_steps: int = 200,
    seed_base: int = 42,
    resume: bool = False,
    db_path: str = DB_PATH,
) -> dict[str, Any]:
    slug = game_id.replace("/", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"human_{slug}.json"

    if resume and out_path.exists():
        print(f"  [skip] {game_id} — already done ({out_path.name})")
        return json.loads(out_path.read_text())

    print(f"\n{'='*60}")
    print(f"  HUMAN-SEEDED: {game_id}  ({n_episodes} episodes)")
    print(f"{'='*60}")

    env = gym.make(game_id, obs_type="ram", render_mode=None)
    n_actions = env.action_space.n
    n_features = 128 + n_actions

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

    # Prioritized replay buffer — seeded from DB
    rng = np.random.RandomState(seed_base)
    buf = PrioritizedReplayBuffer(capacity=50_000, rng=rng)

    n_seeded = buf.seed_from_db(
        db_path=db_path,
        env_name=game_id,
        n_actions=n_actions,
        max_rows=5000,
        verbose=True,
    )

    # Imitation warm-up (only if we have human data)
    if n_seeded > 0 and n_warmup_steps > 0:
        done_wu = imitation_warmup(agent, buf, n_warmup_steps, n_actions)
        print(f"  [warmup] {done_wu} imitation gradient steps done")
    else:
        print(f"  [warmup] Skipped (n_seeded={n_seeded})")

    # RL episodes
    episodes: list[dict] = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_seed = seed_base + ep
        try:
            result = run_episode(env, agent, buf, n_actions, seed=ep_seed)
        except Exception as exc:
            print(f"    ep {ep+1} error: {exc}")
            result = {"reward": 0.0, "steps": 0, "epsilon": agent.epsilon}

        result["episode"] = ep
        result["seeded"] = n_seeded
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

    rewards = [e["reward"] for e in episodes]
    summary = {
        "game":           game_id,
        "arm":            "human_seeded",
        "n_episodes":     n_episodes,
        "n_seeded":       n_seeded,
        "n_warmup_steps": n_warmup_steps,
        "elapsed_s":      round(elapsed, 1),
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "max_reward":     float(np.max(rewards)),
        "min_reward":     float(np.min(rewards)),
        "p25_reward":     float(np.percentile(rewards, 25)),
        "p75_reward":     float(np.percentile(rewards, 75)),
        "episodes":       episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Done. mean={summary['mean_reward']:.2f} "
          f"max={summary['max_reward']:.2f}  seeded={n_seeded}  "
          f"saved → {out_path.name}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Detect played games from DB
# ─────────────────────────────────────────────────────────────────────

def detect_played_games(db_path: str = DB_PATH) -> list[str]:
    import sqlite3
    if not Path(db_path).exists():
        return []
    try:
        con = sqlite3.connect(db_path)
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
        description="Option-A benchmark: RL + human-seeded prioritized replay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--warmup", type=int, default=200,
                   help="Imitation gradient steps before RL (0=skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--db", default=DB_PATH, help="Path to replay_db.sqlite")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    games = args.games or detect_played_games(args.db)
    if not games:
        print("No games found in DB. Play some games first with play_atari_human.py")
        sys.exit(1)

    print(f"\nHuman-seeded benchmark: {len(games)} game(s) x {args.episodes} eps")
    print(f"Warmup: {args.warmup} imitation steps  |  DB: {args.db}\n")

    summaries = []
    for game in games:
        try:
            s = benchmark_game(
                game,
                n_episodes=args.episodes,
                n_warmup_steps=args.warmup,
                seed_base=args.seed,
                resume=args.resume,
                db_path=args.db,
            )
            summaries.append(s)
        except Exception as e:
            print(f"  ERROR on {game}: {e}")

    if summaries:
        print(f"\n{'='*70}")
        print(f"  HUMAN-SEEDED SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Game':<35} {'Mean':>8} {'Std':>7} {'Max':>8} {'Seeded':>7}")
        print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
        for s in summaries:
            name = s['game'].replace('ALE/', '')
            print(f"  {name:<35} {s['mean_reward']:>8.2f} "
                  f"{s['std_reward']:>7.2f} {s['max_reward']:>8.2f} "
                  f"{s.get('n_seeded', 0):>7}")
        combined = RESULTS_DIR / "human_summary.json"
        combined.write_text(json.dumps(summaries, indent=2))
        print(f"\n  Full results → {combined}")
