"""Montezuma runner mode: rehearse."""

from __future__ import annotations

import numpy as np

from brain.games.montezuma.runner.constants import N_ACTIONS
from brain.games.montezuma.runner.setup import make_runner_stack
from brain.games.montezuma.runner.modes.shared import (
    MontezumaStepDeps,
    MontezumaStepState,
    run_montezuma_episode_step,
)


def mode_rehearse(args):
    """
    Focused practice on bottleneck areas.

    Uses save/load state (ALE snapshots) to repeatedly practice
    difficult sections. The rehearsal loop identifies where the
    agent dies most and creates targeted practice sessions.

    Modes:
      advance  — Pause-verify-execute chain building
      frontier — Play from start, advance on death
      stuck    — 10 failures → train flanking areas
      free     — Full exploration with save states
    """
    print("=" * 60)
    print(f"MODE: REHEARSE ({args.rehearse_mode})")
    print("=" * 60)

    stack = make_runner_stack(
        args,
        session_name="montezuma_rehearse",
        render=args.render,
        with_intentional=False,
    )
    env = stack.env
    brain = stack.brain
    modules = stack.modules
    meta_planner = stack.meta_planner
    meta_planner.force_mode("rehearse")

    deps = MontezumaStepDeps(
        env=env,
        brain=brain,
        modules=modules,
        self_model=stack.self_model,
    )

    # First: run a few episodes to identify bottlenecks
    print("\nPhase 1: Identify bottlenecks (10 episodes)...")
    for ep in range(10):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()
        state = MontezumaStepState(
            ram=ram,
            prev_ram=ram.copy(),
            prev_action=0,
            ep_reward=0.0,
            ep_steps=0,
            done=False,
        )

        while not state.done and state.ep_steps < args.max_steps:
            state = run_montezuma_episode_step(
                state, deps, n_actions=N_ACTIONS, use_action_chain=False,
            )

        brain.step(
            state.ram.astype(np.float32) / 255.0,
            prev_action=state.prev_action, reward=state.ep_reward, done=True,
        )

    # Phase 2: Rehearsal
    print(f"\nPhase 2: Rehearse ({args.rehearse_mode} mode, {args.episodes} rounds)...")

    try:
        for round_num in range(args.episodes):
            obs, info = env.reset()
            features = env.unwrapped.ale.getRAM().astype(np.float32) / 255.0
            result = brain.rehearse(mode=args.rehearse_mode, env=env, features=features)
            print(f"  Round {round_num}: {result}")
    except Exception as e:
        print(f"  Rehearsal error: {e}")
        print("  (Rehearsal requires save/load state support in env)")

    env.close()
    brain.close()
