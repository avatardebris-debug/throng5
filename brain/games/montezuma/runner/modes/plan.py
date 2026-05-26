"""Montezuma runner mode: plan."""

from __future__ import annotations

import time

import numpy as np

from brain.games.montezuma.runner.constants import N_ACTIONS
from brain.games.montezuma.runner.setup import make_runner_stack
from brain.games.montezuma.runner.modes.shared import (
    MontezumaStepDeps,
    MontezumaStepState,
    _print_final_report,
    run_montezuma_episode_step,
)


def mode_plan(args):
    """
    Goal-directed training with the SubgoalPlanner active.

    Similar to 'train' but the MetaPlanner is forced into planning
    mode. The agent uses the landmark graph + goal regression +
    causal model to navigate with intentional subgoals rather than
    random exploration.

    Best used AFTER 'ground' mode has discovered RAM semantics.
    """
    print("=" * 60)
    print("MODE: PLAN (goal-directed, SubgoalPlanner active)")
    print("=" * 60)
    print()
    print("MetaPlanner forced to 'planning' mode.")
    print("Agent will use landmark graph + goal regression.")
    print()

    stack = make_runner_stack(
        args,
        session_name="montezuma_plan",
        render=args.render,
        with_intentional=False,
    )
    env = stack.env
    brain = stack.brain
    modules = stack.modules
    meta_planner = stack.meta_planner
    self_model = stack.self_model

    meta_planner.force_mode("planning")

    deps = MontezumaStepDeps(
        env=env,
        brain=brain,
        modules=modules,
        self_model=self_model,
    )

    episode_rewards = []
    best_reward = 0.0
    best_room = 0
    total_steps = 0
    start_time = time.time()

    for episode in range(args.episodes):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()

        state = MontezumaStepState(
            ram=ram,
            prev_ram=ram.copy(),
            prev_action=0,
            ep_reward=0.0,
            ep_steps=0,
            done=False,
            total_steps=total_steps,
        )

        while not state.done and state.ep_steps < args.max_steps:
            state = run_montezuma_episode_step(
                state, deps, n_actions=N_ACTIONS, use_action_chain=False,
            )
            total_steps = state.total_steps

        brain.step(
            state.ram.astype(np.float32) / 255.0,
            prev_action=state.prev_action, reward=state.ep_reward, done=True,
        )

        episode_rewards.append(state.ep_reward)
        best_reward = max(best_reward, state.ep_reward)
        best_room = max(best_room, state.ep_max_room)

        if episode % 10 == 0 or state.ep_reward > 0 or args.verbose:
            avg_r = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1)
            print(f"Ep {episode:4d} | R={state.ep_reward:6.0f} | avg50={avg_r:6.1f} | "
                  f"room={state.ep_max_room}(best={best_room}) | "
                  f"steps={state.ep_steps:5d} | {fps:5.0f} fps | mode=PLANNING")

    elapsed = time.time() - start_time
    _print_final_report(args.episodes, total_steps, best_reward, best_room,
                        episode_rewards, elapsed, modules, meta_planner, self_model)

    env.close()
    brain.close()
