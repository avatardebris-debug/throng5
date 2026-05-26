"""Montezuma env and planning module setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from brain.orchestrator import WholeBrain
from brain.planning.meta_planner import MetaPlanner
from brain.planning.self_model import SelfModel
from brain.planning.ram_semantic_mapper import RAMSemanticMapper
from brain.planning.reward_discovery import RewardDiscovery
from brain.planning.object_graph import ObjectGraph
from brain.planning.safety import SafetyConstraints
from brain.planning.temporal import TemporalReasoner
from brain.planning.procedural_memory import ProceduralMemory
from brain.games.montezuma.runner.constants import GAME_ID, RAM_SIZE
from brain.games.montezuma.runner.game_state import get_game_state

__all__ = ["get_game_state", "GAME_ID", "RAM_SIZE"]


def make_modules():
    """Create all planning modules."""
    ram_mapper = RAMSemanticMapper(ram_size=RAM_SIZE)
    reward_disc = RewardDiscovery(ram_size=RAM_SIZE)
    reward_disc.configure_manual(
        subgoal_bytes=[RAM_ITEMS, RAM_ROOM],
        death_bytes=[RAM_LIVES],
        position_bytes=[RAM_PLAYER_X, RAM_PLAYER_Y],
        item_positions={"skull": (RAM_SKULL_X, RAM_SKULL_Y)},
    )
    return {
        "ram_mapper": ram_mapper,
        "reward_disc": reward_disc,
        "object_graph": ObjectGraph(),
        "safety": SafetyConstraints(),
        "temporal": TemporalReasoner(ram_size=RAM_SIZE),
        "proc_memory": ProceduralMemory(),
    }


def make_env(render: bool = False, frameskip: int = 4):
    """Create the ALE environment."""
    return gym.make(
        GAME_ID,
        frameskip=frameskip,
        render_mode="human" if render else None,
        repeat_action_probability=0.0,
    )


@dataclass
class RunnerStack:
    """Shared Montezuma runner objects built by make_runner_stack()."""
    env: Any
    brain: WholeBrain
    modules: dict
    meta_planner: MetaPlanner
    self_model: SelfModel
    explorer: Optional[Any] = None
    fall_predictor: Optional[Any] = None
    controller: Optional[Any] = None
    action_chain: Optional[Any] = None


def make_runner_stack(
    args,
    *,
    session_name: str,
    render: bool = False,
    with_intentional: bool = True,
    controller_mode: Optional[str] = None,
) -> RunnerStack:
    """
    Factory for env + WholeBrain + planning modules + optional action pipeline.

    Deduplicates setup shared across train/watch/rehearse modes.
    """
    from brain.games.montezuma.runner.constants import N_ACTIONS, RAM_SIZE
    from brain.games.montezuma.runner.exploration import ExplorationManager
    from brain.games.montezuma.runner.fall_predictor import FallPredictor
    from brain.games.montezuma.runner.intentional import IntentionalController
    from brain.games.montezuma.runner.action_policy import ActionFilterChain

    env = make_env(render=render, frameskip=args.frameskip)
    brain = WholeBrain(
        n_features=RAM_SIZE,
        n_actions=N_ACTIONS,
        session_name=session_name,
        use_cnn=False,
        use_torch=True,
    )
    modules = make_modules()
    meta_planner = MetaPlanner(brain)
    self_model = SelfModel(brain)

    explorer = fall_predictor = controller = action_chain = None
    if getattr(brain, "planner", None) is None:
        raise RuntimeError(
            "Montezuma runner requires brain.planner (enable causal_model in "
            "WholeBrain / wire_subsystems; default is enabled)."
        )

    if with_intentional:
        decay = max(getattr(args, "episodes", 100) // 2, 10)
        explorer = ExplorationManager(n_actions=N_ACTIONS, decay_episodes=decay)
        fall_predictor = FallPredictor()
        controller = IntentionalController(
            brain, modules["object_graph"], n_actions=N_ACTIONS,
        )
        if controller_mode is not None:
            controller.set_mode(controller_mode)
        action_chain = ActionFilterChain(
            controller,
            fall_predictor,
            explorer,
            modules["safety"],
            n_actions=N_ACTIONS,
        )

    return RunnerStack(
        env=env,
        brain=brain,
        modules=modules,
        meta_planner=meta_planner,
        self_model=self_model,
        explorer=explorer,
        fall_predictor=fall_predictor,
        controller=controller,
        action_chain=action_chain,
    )
