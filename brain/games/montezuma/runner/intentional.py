"""Closed-loop planning controller for Montezuma."""

from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger(__name__)

# Expected failures from optional counterfactual / Q-value paths.
_CF_EXPECTED = (AttributeError, TypeError, ValueError, RuntimeError, ImportError)

from brain.planning.llm_strategy import LLMStrategy


def _resolve_planning_stack(brain):
    """
    Use WholeBrain.planner from wire_subsystems (single planning graph owner).

    Montezuma runners must use make_runner_stack, which builds WholeBrain with
    causal_model enabled by default.
    """
    planner = getattr(brain, "planner", None)
    if planner is None:
        raise RuntimeError(
            "IntentionalController requires brain.planner. "
            "Use make_runner_stack for Montezuma modes, or enable causal_model in "
            "WholeBrain wire_subsystems (enabled_systems['causal_model']=True)."
        )
    return (
        planner.graph,
        planner.causal_model,
        planner.regressor,
        planner.dead_end_detector,
        planner,
        True,
    )


class IntentionalController:
    """
    Closed-loop planning controller.

    Orchestrates: Grounding → Subgoals → Skills → Reward Shaping → Action.

    Planning stack: reuses brain.planner from wire_subsystems. Causal transition
    observe is delegated to brain.step (planner interval), not duplicated here.

    Flow per step:
      1. Update landmark graph (room change → new landmark)
      2. Generate subgoals if none active (LLMStrategy → SubgoalPlanner)
      3. Shape reward toward current subgoal (distance-based)
      4. Pick action: skill action or DQN + counterfactual
      5. Learn causal effects from transition (local stack only)
    """

    def __init__(self, brain, object_graph, n_actions: int = 18):
        self.brain = brain
        self.object_graph = object_graph
        self.n_actions = n_actions

        (
            self.landmark_graph,
            self.causal_model,
            self.goal_regression,
            self.dead_end_detector,
            self.subgoal_planner,
            self._delegate_causal_observe,
        ) = _resolve_planning_stack(brain)

        self.llm_strategy = LLMStrategy(brain, object_graph=object_graph)
        if brain.skill_library is None:
            raise RuntimeError(
                "IntentionalController requires brain.skill_library. "
                "Enable skill_library in WholeBrain wire_subsystems."
            )
        if brain.counterfactual is None:
            raise RuntimeError(
                "IntentionalController requires brain.counterfactual. "
                "Enable counterfactual in WholeBrain wire_subsystems."
            )
        self.skill_lib = brain.skill_library
        self.counterfactual = brain.counterfactual

        # State
        self._active_skill = None
        self._subgoals = []
        self._current_subgoal_idx = 0
        self._prev_game_state = None
        self._prev_features = None
        self._prev_room = -1
        self._steps_since_plan = 0
        self._replan_interval = 500  # steps between replanning
        self._mode = "reactive"  # reactive, planning, explore

        # Montezuma subgoal targets (room 0 starting knowledge)
        self._known_targets = {
            "key": {"x": 17, "y": 148, "room": 1, "description": "navigate to key"},
            "door": {"x": 133, "y": 148, "room": 1, "description": "navigate to door"},
            "skull": {"x": None, "y": None, "room": 0, "description": "dodge skull"},
        }

        # Stats
        self._subgoals_generated = 0
        self._subgoals_completed = 0
        self._skills_executed = 0
        self._landmarks_found = 0
        self._causal_observations = 0
        self._plans_made = 0

    def set_mode(self, mode: str):
        """Set planning mode: reactive, planning, explore."""
        self._mode = mode

    def on_episode_start(self, game_state: dict):
        """Reset for new episode."""
        self._prev_game_state = game_state
        self._prev_room = game_state["room"]
        self._steps_since_plan = 0
        self._active_skill = None
        self._current_subgoal_idx = 0

    def step(
        self,
        game_state: dict,
        features: np.ndarray,
        brain_action: int,
        reward: float,
        done: bool,
    ) -> dict:
        """
        Full intentional step. Returns:
            action: int
            shaped_reward: float (additional reward to feed brain)
            info: dict with planning state
        """
        self._steps_since_plan += 1
        shaped_reward = 0.0
        action = brain_action
        info = {"source": "dqn", "subgoal": None, "skill": None}

        # ─── 1. Landmark graph: track room changes + items ────────
        if self._prev_game_state is not None:
            if game_state["room"] != self._prev_room:
                # New room = new landmark
                self.landmark_graph.add_landmark(
                    features, label=f"room_{game_state['room']}",
                    step=self._steps_since_plan,
                )
                self._landmarks_found += 1

                # Add edge from previous room
                if self._prev_features is not None:
                    self.landmark_graph.add_edge(
                        self._prev_features, features,
                        actions=[brain_action],
                        confidence=0.5,
                    )
                self._prev_room = game_state["room"]

            # Item collected
            prev_mask = self._prev_game_state.get("items_mask", 0)
            if game_state["items_mask"] != prev_mask:
                self.landmark_graph.add_landmark(
                    features, label=f"item_{game_state['items_mask']:08b}",
                    is_goal=True, step=self._steps_since_plan,
                )
                self._landmarks_found += 1
                shaped_reward += 5.0  # Big bonus for item collection

            # Death tracking
            if done and reward <= 0:
                self.landmark_graph.record_death(features)

        # ─── 2. Causal model: learn effects (skip if brain owns planner) ─
        if not self._delegate_causal_observe and self._prev_features is not None:
            self.causal_model.observe(
                self._prev_features, brain_action, features,
                reward=reward, is_dead_end=(done and reward <= 0),
            )
            self._causal_observations += 1
            self.subgoal_planner.observe_transition(
                self._prev_features, brain_action, features, reward, done,
            )

        # ─── 3. Planning mode: generate subgoals ─────────────────
        if self._mode == "planning":
            needs_plan = (
                not self._subgoals
                or self._steps_since_plan > self._replan_interval
                or (self._active_skill and not self._active_skill._active)
            )

            if needs_plan:
                self._generate_subgoals(game_state, features)

            # ─── 4. Skill execution ──────────────────────────────
            if self._active_skill and self._active_skill._active:
                result = self._active_skill.step(features, game_state, reward)
                skill_action = result.get("action", brain_action)
                status = result.get("status", "running")

                if status == "complete":
                    self._subgoals_completed += 1
                    self._active_skill = None
                    self._current_subgoal_idx += 1
                    # Advance to next subgoal skill
                    self._activate_next_skill(game_state)
                    shaped_reward += 10.0  # Subgoal completion bonus
                elif status in ("failed", "timeout"):
                    self._active_skill = None
                    # Replan
                    self._generate_subgoals(game_state, features)
                else:
                    action = skill_action
                    info["source"] = "skill"
                    info["skill"] = self._active_skill.name if self._active_skill else None

            # ─── 5. Subgoal-directed reward shaping ──────────────
            if self._subgoals and self._current_subgoal_idx < len(self._subgoals):
                sg = self._subgoals[self._current_subgoal_idx]
                target = sg.get("target", {})

                # target can be a string ("skull") or dict ({"x": 17, "y": 148})
                if isinstance(target, dict):
                    tx, ty = target.get("x"), target.get("y")
                else:
                    tx, ty = None, None

                if tx is not None and ty is not None:
                    px, py = game_state["player_x"], game_state["player_y"]
                    dist = abs(px - tx) + abs(py - ty)  # Manhattan distance

                    if self._prev_game_state:
                        prev_dist = (abs(self._prev_game_state["player_x"] - tx)
                                     + abs(self._prev_game_state["player_y"] - ty))
                        # Dense reward: +0.1 per step closer, -0.05 per step farther
                        shaped_reward += (prev_dist - dist) * 0.1

                info["subgoal"] = sg.get("description", "?")

        # ─── 6. Counterfactual action comparison (every 10 steps) ─
        if (self._mode == "planning"
                and self._steps_since_plan % 10 == 0
                and info["source"] == "dqn"):
            cf_action, cf_source = self._safe_counterfactual_action(
                features, brain_action,
            )
            if cf_source is not None:
                action = cf_action
                info["source"] = cf_source

        # Update state
        self._prev_game_state = game_state.copy()
        self._prev_features = features.copy()

        return {
            "action": action,
            "shaped_reward": shaped_reward,
            "info": info,
        }

    def _safe_counterfactual_action(
        self, features: np.ndarray, brain_action: int,
    ):
        """Compare top Q actions via counterfactual; degrade gracefully on failure."""
        try:
            q_vals = self.brain.basal_ganglia.process(
                {"features": features},
            ).get("q_values", None)
            if q_vals is None:
                return brain_action, None
            top_actions = np.argsort(q_vals)[-3:].tolist()
            comparison = self.counterfactual.compare_actions(
                features, top_actions, n_steps=20,
            )
            if comparison.get("regret", 0) > 0.5:
                return comparison.get("best_action", brain_action), "counterfactual"
        except _CF_EXPECTED as exc:
            _log.debug("counterfactual compare: %s", exc)
        except Exception:
            _log.warning("counterfactual compare failed unexpectedly", exc_info=True)
        return brain_action, None

    def _generate_subgoals(self, game_state: dict, features: np.ndarray):
        """Generate subgoals via LLMStrategy or rule-based fallback."""
        self._plans_made += 1
        self._steps_since_plan = 0

        # Update object graph with current entities
        self.object_graph.add_entity("player", {
            "x": game_state["player_x"], "y": game_state["player_y"],
        }, category="player")
        self.object_graph.add_entity("skull", {
            "x": game_state["skull_x"], "y": game_state["skull_y"],
        }, category="enemy")

        # Request plan from LLM (falls back to rule-based)
        plan_result = self.llm_strategy.request_plan(
            goal="reach the next room and collect items",
            context=f"Room {game_state['room']}, "
                    f"pos=({game_state['player_x']},{game_state['player_y']}), "
                    f"items={game_state['items_mask']:08b}, "
                    f"has_key={game_state['has_key']}",
        )

        raw_subgoals = plan_result.get("subgoals", [])

        # If LLM/rule-based gave us subgoals, use them
        if raw_subgoals:
            self._subgoals = raw_subgoals
            self._subgoals_generated += len(raw_subgoals)
        else:
            # Hard-coded Montezuma first-room plan as absolute fallback
            self._subgoals = self._montezuma_fallback_plan(game_state)

        self._current_subgoal_idx = 0

        # Create composite skill from subgoals
        composite = self.skill_lib.from_subgoals(self._subgoals, game_state)
        if composite:
            composite.start()
            self._active_skill = composite
            self._skills_executed += 1

    def _activate_next_skill(self, game_state: dict):
        """Activate the skill for the next subgoal."""
        if self._current_subgoal_idx < len(self._subgoals):
            remaining = self._subgoals[self._current_subgoal_idx:]
            composite = self.skill_lib.from_subgoals(remaining, game_state)
            if composite:
                composite.start()
                self._active_skill = composite
                self._skills_executed += 1

    def _montezuma_fallback_plan(self, game_state: dict) -> list:
        """Hard-coded fallback plan for Montezuma room 0."""
        room = game_state["room"]
        has_key = game_state["has_key"]

        if room == 0 or room == 1:
            if not has_key:
                return [
                    {"description": "dodge skull", "target": "skull"},
                    {"description": "navigate to key", "target": {"x": 17, "y": 148}},
                    {"description": "collect key", "target": "key"},
                    {"description": "navigate to door", "target": {"x": 133, "y": 148}},
                ]
            else:
                return [
                    {"description": "navigate to door", "target": {"x": 133, "y": 148}},
                ]
        else:
            # Unknown room: explore
            return [
                {"description": "navigate to exit", "target": "exit"},
            ]

    def report(self) -> dict:
        return {
            "mode": self._mode,
            "subgoals_generated": self._subgoals_generated,
            "subgoals_completed": self._subgoals_completed,
            "skills_executed": self._skills_executed,
            "landmarks_found": self._landmarks_found,
            "causal_observations": self._causal_observations,
            "plans_made": self._plans_made,
            "landmark_graph": self.landmark_graph.report(),
            "causal_model": self.causal_model.report(),
            "subgoal_planner": self.subgoal_planner.report(),
            "current_subgoal": (
                self._subgoals[self._current_subgoal_idx].get("description", "?")
                if self._subgoals and self._current_subgoal_idx < len(self._subgoals)
                else None
            ),
            "active_skill": self._active_skill.name if self._active_skill else None,
        }
