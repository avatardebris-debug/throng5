"""Composable action filter chain for Montezuma train/watch loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from brain.games.montezuma.runner.exploration import ExplorationManager
from brain.games.montezuma.runner.fall_predictor import FallPredictor
from brain.games.montezuma.runner.intentional import IntentionalController


@dataclass
class ActionStepContext:
    game_state: dict
    features: np.ndarray
    brain_action: int
    ep_reward: float
    done: bool
    n_actions: int


@dataclass
class ActionStepResult:
    action: int
    shaped_reward: float
    ctrl_info: Dict[str, Any]


class ActionFilterChain:
    """
    Ordered action pipeline: IntentionalController → FallPredictor →
    ExplorationManager → SafetyConstraints.
    """

    def __init__(
        self,
        controller: IntentionalController,
        fall_predictor: FallPredictor,
        explorer: ExplorationManager,
        safety,
        *,
        n_actions: int,
        all_actions: Optional[List[int]] = None,
    ):
        self.controller = controller
        self.fall_predictor = fall_predictor
        self.explorer = explorer
        self.safety = safety
        self.n_actions = n_actions
        self._all_actions = all_actions or list(range(n_actions))

    def select(self, ctx: ActionStepContext) -> ActionStepResult:
        ctrl_result = self.controller.step(
            ctx.game_state,
            ctx.features,
            ctx.brain_action,
            reward=ctx.ep_reward,
            done=ctx.done,
        )
        action = ctrl_result["action"]
        shaped_reward = float(ctrl_result.get("shaped_reward", 0.0))

        fall_safe = self.fall_predictor.filter_actions(ctx.game_state, self._all_actions)
        action = self.explorer.select_action(action, ctx.game_state, ctx.features)

        safe_actions = self.safety.filter_actions(
            ctx.features,
            fall_safe,
            player_pos=(ctx.game_state["player_x"], ctx.game_state["player_y"]),
        )
        if action not in safe_actions:
            action = safe_actions[0] if safe_actions else action

        return ActionStepResult(
            action=action,
            shaped_reward=shaped_reward,
            ctrl_info=ctrl_result.get("info", {}),
        )
