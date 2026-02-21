"""
generate_blind_logs.py — Runs short multi-environment episodes and writes
blind trajectory JSON files that offline_generator.py can consume.

Produces one JSON file per environment, using blind_obs (abstract format)
for every step — no game identity leaks.

Usage:
    python generate_blind_logs.py --envs tetris --episodes 3 --steps 80
    python generate_blind_logs.py --envs tetris,atari --episodes 5 --steps 150
    python generate_blind_logs.py --out-dir blind_logs/
"""

import json
import argparse
import random
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.learning.abstract_features import assert_mask_binary


# ---------------------------------------------------------------------------
# Tetris episode runner
# ---------------------------------------------------------------------------

def _run_tetris_episode(level: int, max_steps: int, ep_idx: int,
                        n_episodes: int) -> List[Dict[str, Any]]:
    """Run one Tetris episode and return list of {step, blind_obs, obs} dicts."""
    adapter = TetrisAdapter(level=level, max_pieces=max_steps)
    adapter.reset()

    agent = PortableNNAgent(
        n_features=adapter.n_features,
        config=AgentConfig(epsilon=0.3, n_hidden=64),
    )

    steps = []
    episode_reward = 0.0
    step_idx = 0

    while not adapter.done and step_idx < max_steps:
        valid = adapter.get_valid_actions()
        if not valid:
            break

        action = agent.select_action(valid, adapter.make_features, explore=True)
        _, reward, done, _ = adapter.step(action)
        episode_reward += reward

        # Build abstract feature vector + assert mask integrity (debug mode)
        af = adapter.get_abstract_features(action=action)
        vec = af.to_vector()
        assert_mask_binary(vec, label=f"tetris ep{ep_idx} step{step_idx}")

        blind = adapter.get_blind_obs_str(action=action, reward=reward)

        steps.append({
            "step":       step_idx,
            "blind_obs":  blind,
            # keep old obs for backward compat (not sent to Tetra)
            "obs":        blind,
        })
        step_idx += 1

    print(f"  Ep {ep_idx+1}/{n_episodes}: {step_idx} steps, "
          f"reward={episode_reward:.1f}, "
          f"lines={getattr(adapter.env, 'lines_cleared', 0)}, "
          f"ext_slots={int(af.ext_mask.sum())}")
    return steps


# ---------------------------------------------------------------------------
# Atari episode runner (optional — requires gymnasium[atari])
# ---------------------------------------------------------------------------

def _run_atari_episode(game_id: str, max_steps: int,
                       ep_idx: int, n_episodes: int) -> List[Dict[str, Any]]:
    """Run one Atari episode. Requires gymnasium + ALE ROMs."""
    try:
        from throng4.environments.atari_adapter import AtariAdapter
    except ImportError:
        print("  AtariAdapter not available — skipping Atari episode")
        return []

    try:
        adapter = AtariAdapter(game_id=game_id)
    except Exception as e:
        print(f"  Could not create AtariAdapter for {game_id}: {e}")
        return []

    adapter.reset()
    steps = []
    episode_reward = 0.0

    for step_idx in range(max_steps):
        valid = adapter.get_valid_actions()
        if not valid:
            break

        action = random.choice(valid)
        _, reward, done, _ = adapter.step(action)
        episode_reward += reward

        af = adapter.get_abstract_features(action=action)
        vec = af.to_vector()
        assert_mask_binary(vec, label=f"{game_id} ep{ep_idx} step{step_idx}")

        blind = adapter.get_blind_obs_str(action=action, reward=reward)
        steps.append({"step": step_idx, "blind_obs": blind, "obs": blind})

        if done:
            break

    print(f"  Ep {ep_idx+1}/{n_episodes}: {len(steps)} steps, "
          f"reward={episode_reward:.1f}")
    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(env_spec: str, n_episodes: int, max_steps: int,
             out_dir: Path) -> Path:
    """
    Run n_episodes for env_spec and write a single trajectory JSON.
    Returns path to written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps: List[Dict[str, Any]] = []
    game_id = env_spec

    if env_spec == "tetris":
        print(f"\n=== Environment: tetris (Level 2) ===")
        game_id = "tetris"
        for i in range(n_episodes):
            ep_steps = _run_tetris_episode(
                level=2, max_steps=max_steps, ep_idx=i, n_episodes=n_episodes
            )
            # Offset step indices so they're globally monotonic
            offset = len(all_steps)
            for s in ep_steps:
                s = dict(s, step=s["step"] + offset)
                all_steps.append(s)

    elif env_spec.startswith("atari:"):
        game_id = env_spec[len("atari:"):]
        print(f"\n=== Environment: {game_id} ===")
        for i in range(n_episodes):
            ep_steps = _run_atari_episode(
                game_id=game_id, max_steps=max_steps,
                ep_idx=i, n_episodes=n_episodes
            )
            offset = len(all_steps)
            for s in ep_steps:
                s = dict(s, step=s["step"] + offset)
                all_steps.append(s)

    else:
        print(f"Unknown env spec: {env_spec!r}. Use 'tetris' or 'atari:GameName'")
        return None

    out = {
        "game_id":    game_id,
        "episodes":   n_episodes,
        "trajectory": all_steps,
    }

    safe = game_id.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"blind_traj_{safe}_{n_episodes}ep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n✅ Written: {out_path}  ({len(all_steps)} steps total)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate blind trajectory logs for offline_generator"
    )
    parser.add_argument("--envs", default="tetris",
                        help="Comma-separated env specs: tetris,atari:GameName")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per environment")
    parser.add_argument("--steps", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--out-dir", default="blind_logs",
                        help="Output directory for trajectory JSONs")
    parser.add_argument("--process", action="store_true",
                        help="Immediately run offline_generator on output files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    env_specs = [e.strip() for e in args.envs.split(",") if e.strip()]

    written = []
    for spec in env_specs:
        p = generate(spec, args.episodes, args.steps, out_dir)
        if p:
            written.append(p)

    if args.process and written:
        print("\n=== Running offline_generator on all outputs ===")
        from throng4.llm_policy.offline_generator import OfflineGenerator
        for log_path in written:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            game_id = data["game_id"]
            gen = OfflineGenerator(game_id=game_id)
            print(f"\n--- Processing {log_path.name} ---")
            print(f"  game_id:     {game_id}")
            print(f"  blind_label: {gen.blind_label}")
            gen.process_log(str(log_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
