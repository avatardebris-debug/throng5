"""Montezuma runner mode: watch."""

from __future__ import annotations

import numpy as np

from brain.games.montezuma.runner.constants import N_ACTIONS
from brain.games.montezuma.runner.setup import get_game_state, make_runner_stack
from brain.games.montezuma.runner.modes.shared import (
    MontezumaStepDeps,
    MontezumaStepState,
    run_montezuma_episode_step,
)


def mode_watch(args):
    """
    Watch the agent play with rendering.

    Now with full intentional planning: subgoals, skills, reward shaping,
    and counterfactual action comparison.
    """
    print("=" * 60)
    print("MODE: WATCH AGENT PLAY (Intentional)")
    print("=" * 60)

    stack = make_runner_stack(
        args,
        session_name="montezuma_watch",
        render=True,
        with_intentional=True,
        controller_mode="planning",
    )
    env = stack.env
    brain = stack.brain
    modules = stack.modules
    meta_planner = stack.meta_planner
    self_model = stack.self_model
    explorer = stack.explorer
    fall_predictor = stack.fall_predictor
    controller = stack.controller
    action_chain = stack.action_chain

    for episode in range(args.episodes):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()
        explorer.on_episode_start()
        fall_predictor.on_episode_start()
        game_state = get_game_state(ram)
        controller.on_episode_start(game_state)

        state = MontezumaStepState(
            ram=ram,
            prev_ram=ram.copy(),
            prev_action=0,
            ep_reward=0.0,
            ep_steps=0,
            done=False,
        )
        deps = MontezumaStepDeps(
            env=env,
            brain=brain,
            modules=modules,
            meta_planner=meta_planner,
            self_model=self_model,
            explorer=explorer,
            fall_predictor=fall_predictor,
            action_chain=action_chain,
        )
        last_subgoal = None

        def _log_subgoal(ctrl_info):
            nonlocal last_subgoal
            sg = ctrl_info.get("subgoal")
            if sg != last_subgoal and sg:
                print(f"  🎯 Subgoal: {sg} (via {ctrl_info['source']})")
                last_subgoal = sg

        while not state.done and state.ep_steps < args.max_steps:
            state = run_montezuma_episode_step(
                state, deps, n_actions=N_ACTIONS, use_action_chain=True,
                track_room=False, on_subgoal_change=_log_subgoal,
            )

        brain.step(
            state.ram.astype(np.float32) / 255.0,
            prev_action=state.prev_action, reward=state.ep_reward, done=True,
        )

        mode = meta_planner.decide()
        if mode in ("planning", "explore"):
            controller.set_mode(mode)
        elif mode == "reactive":
            controller.set_mode("reactive")

        ctrl_rpt = controller.report()
        gs = get_game_state(state.ram)
        print(f"Ep {episode:3d} | R={state.ep_reward:6.0f} | "
              f"room={gs['room']} | steps={state.ep_steps} | "
              f"mode={mode} | ε={explorer.epsilon:.2f} | "
              f"landmarks={ctrl_rpt['landmarks_found']} | "
              f"subgoals={ctrl_rpt['subgoals_completed']}/{ctrl_rpt['subgoals_generated']} | "
              f"causal={ctrl_rpt['causal_observations']}")

    env.close()
    brain.close()
