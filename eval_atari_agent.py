"""
eval_atari_agent.py
===================
Option 1 evaluation runner: loads trained agent weights, runs
evaluation episodes with REAL Q-values, decodes RAM state (if available),
and writes canonical JSONL events for Tetra analysis.

This is the "right long-term" path:
    Human plays  →  replay_db (raw)
    replay_db    →  Imitation training (overnight)
    THIS SCRIPT  →  Agent plays + logs → atari_brief.json
    brief        →  Tetra  →  hypotheses  →  next training run

Two modes
---------
1. Trained agent  (--weights path/to/weights.npz)
   Agent has learned from human data; events show how the human strategy
   was absorbed and where gaps remain.

2. Baseline agent (no --weights flag, or --baseline)
   Uninformed agent; events show pure-RL behaviour for Tetra to contrast.

Key difference from backfill
-----------------------------
Events use REAL softmax(Q-values) — not estimated confidence.
entropy, agent_topk, and calibration_proxy are genuine.

Usage
-----
    # Evaluate imitation-trained agent on Montezuma (overnight weights)
    python eval_atari_agent.py --game ALE/MontezumaRevenge-v5 \\
        --weights benchmark_results/human_ALE_MontezumaRevenge_v5_weights.npz \\
        --episodes 10

    # Baseline comparison
    python eval_atari_agent.py --game ALE/Breakout-v5 --baseline --episodes 5

    # All games with trained weights (batch mode)
    python eval_atari_agent.py --all --weights-dir benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from throng4.storage.atari_event import AtariEventLogger, update_brief
from throng4.storage.ram_decoders import get_decoder
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig

_WEIGHTS_DIR = _ROOT / "benchmark_results"
_SESSION_PREFIX = "eval"


# ─────────────────────────────────────────────────────────────────────
# Agent wrapper — consistent with play_atari_human.py _AgentVoter
# ─────────────────────────────────────────────────────────────────────

def _make_agent(n_actions: int, weights_path: Optional[str] = None) -> PortableNNAgent:
    n_features = 128 + n_actions   # RAM (128) + one-hot action
    cfg = AgentConfig(
        n_hidden=256, n_hidden2=128,
        epsilon=0.0,  # eval mode: greedy
        use_imitation_head=True,
        imitation_n_actions=n_actions,
    )
    agent = PortableNNAgent(n_features, config=cfg, seed=42)
    if weights_path:
        try:
            agent.load_weights(weights_path)
            print(f"  [agent] Loaded weights: {weights_path}")
        except Exception as exc:
            print(f"  [agent] WARNING: could not load weights ({exc}). Using random init.")
    return agent


def _q_values_all(agent: PortableNNAgent, ram_obs: np.ndarray,
                  n_actions: int) -> np.ndarray:
    """Compute Q(s,a) for every action via separate forward passes."""
    state = np.array(ram_obs, dtype=np.float32) / 255.0
    q = np.empty(n_actions, dtype=np.float64)
    for a in range(n_actions):
        ah = np.zeros(n_actions, dtype=np.float32)
        ah[a] = 1.0
        q[a] = agent.forward(np.concatenate([state, ah]))
    return q


# ─────────────────────────────────────────────────────────────────────
# Single-game evaluation
# ─────────────────────────────────────────────────────────────────────

def eval_game(
    game_id: str,
    n_episodes: int = 10,
    weights_path: Optional[str] = None,
    seed: int = 0,
    label: str = "eval",        # "eval" or "baseline" — used in session ID
    verbose: bool = True,
) -> Path:
    """
    Run n_episodes of evaluation and write JSONL events.
    Returns path to the JSONL file.
    """
    env = gym.make(game_id, obs_type="ram", render_mode=None)
    ram_obs, _ = env.reset(seed=seed)

    n_actions = env.action_space.n
    action_meanings = list(env.unwrapped.get_action_meanings())
    decoder = get_decoder(game_id)

    if verbose:
        print(f"\n[{label}] {game_id}  n_actions={n_actions}")
        if decoder:
            print(f"  [decoder] {type(decoder).__name__} available")
        else:
            print(f"  [decoder] None — raw RAM only")

    agent = _make_agent(n_actions, weights_path)

    session_id = f"{label}_{game_id.replace('/', '_').replace('-', '_')}_ep{n_episodes}"
    evt_logger = AtariEventLogger(game_id, action_meanings, session_id)

    # We also maintain a separate semantic log (RAM-decoded) per episode
    semantic_episodes = []

    for ep in range(n_episodes):
        ram_obs, _ = env.reset(seed=seed + ep)
        evt_logger.begin_episode(ep)

        step = 0
        total_reward = 0.0
        done = False
        ep_decoded_steps = []

        while not done:
            q_vals = _q_values_all(agent, ram_obs, n_actions)
            agent_action = int(np.argmax(q_vals))

            # Log canonical event (Q-values are REAL, not reconstructed)
            evt_logger.log_step(
                step=step,
                human_action_idx=agent_action,   # agent IS the "human" here
                q_values=q_vals,
                reward=total_reward,
                done=done,
                near_death=False,
            )

            # Semantic decode (if decoder available)
            if decoder:
                decoded = decoder.decode(ram_obs)
                decoded["step"] = step
                decoded["q_entropy"] = float(-np.sum(
                    np.exp(q_vals - q_vals.max()) /
                    np.exp(q_vals - q_vals.max()).sum() *
                    np.log(np.exp(q_vals - q_vals.max()) /
                           np.exp(q_vals - q_vals.max()).sum() + 1e-9)
                ))
                decoded["top_action"] = action_meanings[agent_action]
                ep_decoded_steps.append(decoded)

            ram_obs, reward, term, trunc, _ = env.step(agent_action)
            total_reward += reward
            done = term or trunc
            step += 1

        semantic_episodes.append({
            "episode": ep,
            "total_reward": total_reward,
            "total_steps": step,
            "decoded_steps": ep_decoded_steps,
        })

        if verbose:
            rooms_visited = (
                sorted({s.get("room") for s in ep_decoded_steps})
                if ep_decoded_steps else "N/A"
            )
            keys = sum(1 for s in ep_decoded_steps if s.get("key_collected", False))
            print(f"  ep {ep:>3}: reward={total_reward:+.1f}  steps={step:>4}"
                  f"  rooms={rooms_visited}  key_steps={keys}")

    env.close()

    # Close event logger
    evt_path = evt_logger.end_session()
    if verbose:
        print(f"  events  -> {evt_path}")

    # Write semantic log (decoded state per step)
    if any(ep["decoded_steps"] for ep in semantic_episodes):
        sem_path = evt_path.with_name(evt_path.stem + "_semantic.json")
        sem_out = {
            "game":     game_id,
            "label":    label,
            "decoder":  type(decoder).__name__ if decoder else None,
            "episodes": [
                {
                    "episode":       ep["episode"],
                    "total_reward":  ep["total_reward"],
                    "total_steps":   ep["total_steps"],
                    # Summarise: don't write full step-by-step (too large)
                    "rooms_visited": sorted({
                        s["room"] for s in ep["decoded_steps"]
                        if "room" in s
                    }),
                    "max_room":   max(
                        (s["room"] for s in ep["decoded_steps"] if "room" in s),
                        default=0,
                    ),
                    "key_collected_any": any(
                        s.get("key_collected", False)
                        for s in ep["decoded_steps"]
                    ),
                    "mean_entropy": round(
                        float(np.mean([s["q_entropy"] for s in ep["decoded_steps"]])), 4
                    ),
                    # Snapshot every 50 steps for Tetra to read the trajectory
                    "trajectory_snapshots": [
                        s for s in ep["decoded_steps"]
                        if s["step"] % 50 == 0
                    ][:10],
                }
                for ep in semantic_episodes
            ],
        }
        sem_path.write_text(json.dumps(sem_out, indent=2))
        if verbose:
            print(f"  semantic -> {sem_path}")

    # Update Tetra brief
    brief_path = update_brief(game_id)
    if verbose:
        print(f"  brief   -> {brief_path}")

    return evt_path


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Run trained Atari agent for evaluation + Tetra logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game", default="ALE/MontezumaRevenge-v5",
                   help="Game to evaluate")
    p.add_argument("--all", action="store_true",
                   help="Evaluate all games that have trained weights in --weights-dir")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--weights", default=None,
                   help="Path to .npz agent weights (omit for random/baseline)")
    p.add_argument("--weights-dir", default=str(_WEIGHTS_DIR),
                   help="Directory to search for weights when --all is used")
    p.add_argument("--baseline", action="store_true",
                   help="Run as baseline (no weights, label='baseline')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    label = "baseline" if args.baseline else "eval"

    if args.all:
        # Find all weights files in weights-dir
        wdir = Path(args.weights_dir)
        weight_files = list(wdir.glob("human_*_weights.npz"))
        if not weight_files:
            print(f"No *_weights.npz found in {wdir}")
            sys.exit(1)

        for wf in sorted(weight_files):
            # Reconstruct game_id from filename: human_ALE_Game_v5_weights.npz
            slug = wf.stem.replace("human_", "").replace("_weights", "")
            # slug = ALE_MontezumaRevenge_v5 -> ALE/MontezumaRevenge-v5
            parts = slug.split("_")
            game_id = parts[0] + "/" + "_".join(parts[1:-1]) + "-" + parts[-1]
            try:
                eval_game(game_id, args.episodes, str(wf),
                          seed=args.seed, label=label,
                          verbose=not args.quiet)
            except Exception as exc:
                print(f"  ERROR {game_id}: {exc}")
    else:
        eval_game(
            args.game,
            args.episodes,
            args.weights,
            seed=args.seed,
            label=label,
            verbose=not args.quiet,
        )
