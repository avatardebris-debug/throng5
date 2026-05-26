"""Montezuma runner mode: train."""

from __future__ import annotations

import json
import time

import numpy as np

from brain.environments.human_recorder import HumanRecorder
from brain.games.montezuma.runner.constants import (
    N_ACTIONS, RESULTS_DIR, RECORDINGS_DIR, ensure_results_dirs,
)
from brain.games.montezuma.runner.setup import get_game_state, make_runner_stack
from brain.games.montezuma.runner.modes.shared import (
    MontezumaStepDeps,
    MontezumaStepState,
    _print_final_report,
    _save_stats,
    run_montezuma_episode_step,
)


def mode_train(args):
    """
    Standard training mode. No rendering, maximum speed.

    Runs the full brain stack with all planning modules active.
    Meta-planner auto-selects mode based on performance:
      reactive → planning → rehearse → LLM consult
    """
    print("=" * 60)
    print("MODE: TRAIN (max speed, all modules active)")
    print("=" * 60)

    stack = make_runner_stack(
        args,
        session_name="montezuma_train",
        render=False,
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
    recorder = HumanRecorder("train") if args.record else None

    grounding_path = RESULTS_DIR / "grounding.json"
    if grounding_path.exists():
        print(f"Loading grounding from {grounding_path}")
        with open(grounding_path) as f:
            grounding = json.load(f)
        print(f"  {grounding.get('frames_analyzed', '?')} frames analyzed")

    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps/episode: {args.max_steps}")
    print()

    episode_rewards = []
    best_reward = 0.0
    best_room = 0
    total_steps = 0
    start_time = time.time()

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
            total_steps=total_steps,
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
            recorder=recorder,
        )

        if recorder:
            recorder.start(None)

        while not state.done and state.ep_steps < args.max_steps:
            state = run_montezuma_episode_step(
                state, deps, n_actions=N_ACTIONS, use_action_chain=True, track_room=True,
            )
            total_steps = state.total_steps

        brain.step(
            state.ram.astype(np.float32) / 255.0,
            prev_action=state.prev_action, reward=state.ep_reward, done=True,
        )
        meta_planner.observe(
            state.ram.astype(np.float32) / 255.0, reward=state.ep_reward, done=True,
        )

        if recorder:
            recorder.stop()

        episode_rewards.append(state.ep_reward)
        best_reward = max(best_reward, state.ep_reward)
        best_room = max(best_room, state.ep_max_room)

        mode = meta_planner.decide()
        if mode in ("planning", "explore"):
            controller.set_mode(mode)
        elif mode == "reactive":
            controller.set_mode("reactive")

        elapsed = time.time() - start_time
        fps = total_steps / max(elapsed, 1)

        if episode % 10 == 0 or state.ep_reward > 0 or args.verbose:
            avg_r = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            ctrl_rpt = controller.report()
            print(f"Ep {episode:4d} | R={state.ep_reward:6.0f} | avg50={avg_r:6.1f} | "
                  f"best={best_reward:6.0f} | room={state.ep_max_room}(best={best_room}) | "
                  f"steps={state.ep_steps:5d} | intr={state.ep_intrinsic:6.2f} | "
                  f"mode={mode:12s} | ε={explorer.epsilon:.2f} | "
                  f"sg={ctrl_rpt['subgoals_completed']}/{ctrl_rpt['subgoals_generated']} | "
                  f"{fps:5.0f} fps")

        if (episode + 1) % args.save_freq == 0:
            _save_stats(episode, total_steps, best_reward, best_room,
                        episode_rewards, elapsed, mode, modules, meta_planner,
                        self_model, brain)

    elapsed = time.time() - start_time
    _print_final_report(args.episodes, total_steps, best_reward, best_room,
                        episode_rewards, elapsed, modules, meta_planner, self_model)

    if recorder:
        ensure_results_dirs()
        rec_path = recorder.save(str(RECORDINGS_DIR))
        print(f"  Training recording saved to {rec_path}")

    env.close()
    brain.close()
