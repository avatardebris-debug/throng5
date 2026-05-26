"""Shared helpers for Montezuma runner modes."""

from __future__ import annotations

import json

import logging

from dataclasses import dataclass

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from brain.games.montezuma.runner.constants import RESULTS_DIR, ensure_results_dirs

from brain.games.montezuma.runner.setup import get_game_state

from brain.games.montezuma.runner.action_policy import (
    ActionStepContext,
    ActionFilterChain,
)

_log = logging.getLogger(__name__)


@dataclass
class MontezumaStepState:
    """Mutable per-episode loop state for train/watch modes."""

    ram: np.ndarray
    prev_ram: np.ndarray
    prev_action: int
    ep_reward: float
    ep_steps: int
    done: bool
    ep_max_room: int = 0
    ep_intrinsic: float = 0.0
    total_steps: int = 0


@dataclass
class MontezumaStepDeps:

    env: Any
    brain: Any
    modules: dict
    self_model: Any
    meta_planner: Optional[Any] = None
    explorer: Optional[Any] = None
    fall_predictor: Optional[Any] = None
    action_chain: Optional[ActionFilterChain] = None
    recorder: Optional[Any] = None


def _observe_modules(
    state: MontezumaStepState,
    deps: MontezumaStepDeps,
    ram: np.ndarray,
    *,
    use_action_chain: bool,
) -> Dict[str, Any]:
    """Read RAM, update mapping modules, return semantic game_state."""
    game_state = get_game_state(ram)
    deps.modules["ram_mapper"].observe(
        ram,
        action=state.prev_action,
        reward=state.ep_reward,
        done=state.done,
    )
    if use_action_chain:
        deps.modules["temporal"].observe(ram, step=state.ep_steps)
    return game_state


def _compute_intrinsic_reward(
    state: MontezumaStepState,
    deps: MontezumaStepDeps,
    ram: np.ndarray,
    game_state: Dict[str, Any],
    *,
    use_action_chain: bool,
) -> float:

    intrinsic_r = deps.modules["reward_disc"].compute(
        state.prev_ram,
        ram,
        action=state.prev_action,
        done=state.done,
    )
    if use_action_chain and deps.explorer is not None:
        intrinsic_r += deps.explorer.get_novelty_bonus(game_state)
        state.ep_intrinsic += intrinsic_r
        deps.modules["object_graph"].add_entity(
            "player",
            {
                "x": game_state["player_x"],
                "y": game_state["player_y"],
            },
            category="player",
        )
        deps.modules["object_graph"].add_entity(
            "skull",
            {
                "x": game_state["skull_x"],
                "y": game_state["skull_y"],
            },
            category="enemy",
        )
    return intrinsic_r


def _brain_step(
    deps: MontezumaStepDeps,
    ram: np.ndarray,
    state: MontezumaStepState,
    intrinsic_r: float,
) -> Tuple[np.ndarray, int]:

    features = ram.astype(np.float32) / 255.0
    result = deps.brain.step(
        features,
        prev_action=state.prev_action,
        reward=intrinsic_r,
        done=False,
    )
    brain_action = result["action"] if isinstance(result, dict) else result
    return features, brain_action


def _select_action(
    state: MontezumaStepState,
    deps: MontezumaStepDeps,
    game_state: Dict[str, Any],
    features: np.ndarray,
    brain_action: int,
    intrinsic_r: float,
    *,
    n_actions: int,
    use_action_chain: bool,
    on_subgoal_change: Optional[Callable[[Dict[str, Any]], None]],
) -> Tuple[int, float, Dict[str, Any]]:

    if use_action_chain:
        if deps.action_chain is None:
            raise RuntimeError(
                "use_action_chain=True requires MontezumaStepDeps.action_chain "
                "(make_runner_stack with_intentional=True)"
            )
        policy_result = deps.action_chain.select(
            ActionStepContext(
                game_state=game_state,
                features=features,
                brain_action=brain_action,
                ep_reward=state.ep_reward,
                done=state.done,
                n_actions=n_actions,
            )
        )
        action = policy_result.action
        intrinsic_r += policy_result.shaped_reward
        ctrl_info = policy_result.ctrl_info
        if on_subgoal_change is not None:
            on_subgoal_change(ctrl_info)
        if deps.meta_planner is not None:
            deps.meta_planner.observe(
                features, reward=intrinsic_r, done=False, action=action
            )
        deps.modules["proc_memory"].observe_transition(
            action,
            features,
            features,
            intrinsic_r,
        )
        if deps.recorder:
            deps.recorder.record(state.ram, action, state.ep_reward, False)
        return action, intrinsic_r, ctrl_info
    safe_actions = deps.modules["safety"].filter_actions(
        features,
        list(range(n_actions)),
        player_pos=(game_state["player_x"], game_state["player_y"]),
    )
    action = brain_action
    if action not in safe_actions:
        action = safe_actions[0] if safe_actions else action
    return action, intrinsic_r, {}


def _env_step(
    deps: MontezumaStepDeps,
    action: int,
    state: MontezumaStepState,
) -> float:

    obs, reward, terminated, truncated, info = deps.env.step(action)
    state.done = terminated or truncated
    state.ep_reward += reward
    state.ep_steps += 1
    state.total_steps += 1
    state.prev_action = action
    return float(reward)


def _post_step(
    state: MontezumaStepState,
    deps: MontezumaStepDeps,
    ram: np.ndarray,
    game_state: Dict[str, Any],
    features: np.ndarray,
    action: int,
    step_reward: float,
    *,
    use_action_chain: bool,
) -> None:

    state.prev_ram = ram.copy()
    state.ram = ram
    if use_action_chain and deps.fall_predictor is not None:
        deps.fall_predictor.observe(
            get_game_state(deps.env.unwrapped.ale.getRAM()),
            action,
        )
    deps.self_model.record_position(
        game_state["player_x"],
        game_state["player_y"],
        area_hash=game_state["room"],
    )
    if state.done and step_reward <= 0:
        if use_action_chain:
            deps.modules["safety"].learn_from_death(
                features,
                state.prev_action,
                context="death",
            )
            deps.self_model.record_death("death", features)
        else:
            deps.modules["safety"].learn_from_death(features, state.prev_action)
            deps.self_model.record_death("death")


def run_montezuma_episode_step(
    state: MontezumaStepState,
    deps: MontezumaStepDeps,
    *,
    n_actions: int,
    use_action_chain: bool = True,
    track_room: bool = True,
    on_subgoal_change: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MontezumaStepState:
    """
    One inner-loop iteration: observe RAM, brain step, action selection, env step.
    use_action_chain=True (train/watch): ActionFilterChain + exploration modules.
    use_action_chain=False (plan/rehearse bottleneck): brain action + safety filter only.
    """
    ram = deps.env.unwrapped.ale.getRAM()
    game_state = _observe_modules(state, deps, ram, use_action_chain=use_action_chain)
    if track_room or not use_action_chain:
        state.ep_max_room = max(state.ep_max_room, game_state["room"])
    intrinsic_r = _compute_intrinsic_reward(
        state,
        deps,
        ram,
        game_state,
        use_action_chain=use_action_chain,
    )
    features, brain_action = _brain_step(deps, ram, state, intrinsic_r)
    action, intrinsic_r, _ctrl_info = _select_action(
        state,
        deps,
        game_state,
        features,
        brain_action,
        intrinsic_r,
        n_actions=n_actions,
        use_action_chain=use_action_chain,
        on_subgoal_change=on_subgoal_change,
    )
    step_reward = _env_step(deps, action, state)
    _post_step(
        state,
        deps,
        ram,
        game_state,
        features,
        action,
        step_reward,
        use_action_chain=use_action_chain,
    )
    return state


def _save_stats(
    episode,
    total_steps,
    best_reward,
    best_room,
    episode_rewards,
    elapsed,
    mode,
    modules,
    meta_planner,
    self_model,
    brain,
):

    stats = {
        "episode": episode,
        "total_steps": total_steps,
        "best_reward": best_reward,
        "best_room": best_room,
        "avg_reward_50": float(np.mean(episode_rewards[-50:])),
        "elapsed_seconds": round(elapsed, 1),
        "meta_mode": mode,
        "ram_mapper": modules["ram_mapper"].report(),
        "reward_discovery": modules["reward_disc"].report(),
        "safety": modules["safety"].report(),
        "temporal": modules["temporal"].report(),
        "self_model": self_model.report(),
        "meta_planner": meta_planner.report(),
        "proc_memory": modules["proc_memory"].report(),
        "brain": brain.report(),
    }
    ensure_results_dirs()
    stats_path = RESULTS_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def _print_final_report(
    n_episodes,
    total_steps,
    best_reward,
    best_room,
    episode_rewards,
    elapsed,
    modules,
    meta_planner,
    self_model,
):

    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"  Episodes: {n_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Best reward: {best_reward:.0f}")
    print(f"  Best room: {best_room}")
    print(f"  Avg reward (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  Time: {elapsed:.0f}s ({total_steps/max(elapsed,1):.0f} fps)")
    print()
    print(f"  RAM Mapper: {modules['ram_mapper'].report()}")
    print(f"  Reward Discovery: {modules['reward_disc'].report()}")
    print(f"  Safety: {modules['safety'].report()}")
    print(f"  Procedural Memory: {modules['proc_memory'].report()}")
    print(f"  Meta-Planner: {meta_planner.report()}")
    print(f"  Self-Model: {self_model.report()}")
    print("=" * 60)
