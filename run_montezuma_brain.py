"""
run_montezuma_brain.py — Montezuma's Revenge with the WholeBrain stack.

Thin CLI entrypoint; implementation in brain.games.montezuma.runner.
"""

from __future__ import annotations

import argparse
import warnings

import ale_py
import gymnasium as gym

from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ale_py.register_v5_envs()

from brain.games.montezuma.runner.modes import (
    mode_ground,
    mode_human,
    mode_plan,
    mode_rehearse,
    mode_train,
    mode_watch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Montezuma's Revenge — Full WholeBrain Stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  human     Play yourself with keyboard, record RAM for grounding
  watch     Watch the agent play (rendered)
  ground    Analyze a human recording to discover RAM semantics
  train     Standard training (no rendering, max speed)
  plan      Goal-directed training with SubgoalPlanner
  rehearse  Focused bottleneck practice with save states
""",
    )
    parser.add_argument(
        "mode",
        choices=["human", "watch", "ground", "train", "plan", "rehearse"],
        help="Operating mode",
    )
    parser.add_argument(
        "recording",
        nargs="?",
        default=None,
        help="Path to recording file (for 'ground' mode)",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-freq", type=int, default=25)
    parser.add_argument(
        "--rehearse-mode",
        choices=["advance", "frontier", "stuck", "free"],
        default="advance",
    )
    args = parser.parse_args()

    dispatch = {
        "human": mode_human,
        "watch": mode_watch,
        "ground": mode_ground,
        "train": mode_train,
        "plan": mode_plan,
        "rehearse": mode_rehearse,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
