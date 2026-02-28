"""
subgoal_planner.py — Execute multi-step plans via subgoal decomposition.

The Manager/Worker pattern:
  Manager (this module): selects which subgoal to pursue next
  Worker (Striatum): executes actions to achieve current subgoal

The Manager operates at a coarser timescale (every 50-100 steps) while
the Worker runs every step. This enables long-term planning while
keeping the reactive loop fast.

Usage:
    planner = SubgoalPlanner(brain, graph, regressor)
    plan = planner.make_plan(current_features, goal_features)
    action = planner.get_action(current_features)  # Manager-guided
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

import numpy as np

from brain.planning.landmark_graph import LandmarkGraph
from brain.planning.goal_regression import GoalRegression, Subgoal
from brain.planning.dead_end_detector import DeadEndDetector
from brain.planning.causal_model import CausalModel


class SubgoalPlanner:
    """
    Hierarchical goal-directed planner (Manager).

    Maintains a plan (ordered subgoals), tracks progress, and
    replans when subgoals fail or dead ends are detected.
    """

    def __init__(
        self,
        brain,
        graph: LandmarkGraph,
        regressor: GoalRegression,
        dead_end_detector: DeadEndDetector,
        causal_model: CausalModel,
        subgoal_timeout: int = 200,     # Steps before declaring subgoal failed
        replan_on_failure: bool = True,
    ):
        self.brain = brain
        self.graph = graph
        self.regressor = regressor
        self.dead_end_detector = dead_end_detector
        self.causal_model = causal_model
        self.subgoal_timeout = subgoal_timeout
        self.replan_on_failure = replan_on_failure

        # Current plan state
        self._plan: List[Subgoal] = []
        self._current_subgoal_idx: int = 0
        self._steps_on_current: int = 0
        self._goal_hash: Optional[int] = None
        self._achieved: Set[int] = set()

        # Stats
        self._plans_made: int = 0
        self._plans_completed: int = 0
        self._plans_failed: int = 0
        self._subgoals_achieved: int = 0
        self._subgoals_failed: int = 0
        self._dead_ends_caught: int = 0
        self._traps_caught: int = 0

    @property
    def has_plan(self) -> bool:
        return len(self._plan) > 0

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        if self._current_subgoal_idx < len(self._plan):
            return self._plan[self._current_subgoal_idx]
        return None

    @property
    def plan_progress(self) -> float:
        if not self._plan:
            return 0.0
        return self._current_subgoal_idx / len(self._plan)

    # ── Plan Creation ────────────────────────────────────────────────

    def make_plan(
        self,
        current_features: np.ndarray,
        goal_features: np.ndarray,
        goal_label: str = "goal",
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Create a plan from current state to goal.

        Uses GoalRegression (backward chaining) with forward Dijkstra fallback.
        Returns the plan as a list of subgoal dicts, or None if impossible.
        """
        self._plans_made += 1

        current_hash = self.graph.add_landmark(current_features, label="current")
        goal_hash = self.graph.add_landmark(goal_features, label=goal_label, is_goal=True)
        self._goal_hash = goal_hash

        # Try backward chaining first
        plan = self.regressor.regress(goal_hash, current_hash)

        if plan is None:
            # Try forward Dijkstra
            route = self.graph.plan_route(current_features, goal_features)
            if route:
                plan = [
                    Subgoal(state_hash=step["to"], label=f"waypoint_{i}")
                    for i, step in enumerate(route)
                ]

        if plan is None:
            self._plans_failed += 1
            return None

        self._plan = plan
        self._current_subgoal_idx = 0
        self._steps_on_current = 0
        self._achieved = set()

        return [sg.to_dict() for sg in plan]

    # ── Execution (Manager) ──────────────────────────────────────────

    def get_action(
        self,
        features: np.ndarray,
        default_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get the Manager's recommended action.

        Returns dict with:
            action: int — recommended action
            source: str — "plan", "worker", "explore"
            subgoal: current subgoal info
            plan_status: overall plan status
        """
        if not self.has_plan:
            return {
                "action": default_action or 0,
                "source": "no_plan",
                "subgoal": None,
                "plan_status": "no_plan",
            }

        current_sg = self.current_subgoal
        if current_sg is None:
            self._plans_completed += 1
            return {
                "action": default_action or 0,
                "source": "plan_complete",
                "subgoal": None,
                "plan_status": "complete",
            }

        self._steps_on_current += 1

        # Check if current subgoal is achieved
        current_hash = self.graph._hash_state(features)
        if current_hash == current_sg.state_hash:
            return self._advance_subgoal(features, default_action)

        # Check for timeout
        if self._steps_on_current > self.subgoal_timeout:
            return self._handle_subgoal_failure(features, default_action)

        # Check for dead end
        if self._steps_on_current % 50 == 0:
            dead_end_info = self._check_dead_end(features)
            if dead_end_info.get("is_dead_end"):
                return self._handle_dead_end(features, default_action)

        # Check for dangerous actions via causal model
        n_actions = getattr(self.brain.striatum, '_n_actions', 4)
        safe_actions = self.causal_model.get_safe_actions(features, n_actions)

        # Get route to current subgoal
        sg_lm = self.graph.get_landmark(current_sg.state_hash)
        if sg_lm and sg_lm.features is not None:
            route = self.graph.plan_route(features, sg_lm.features)
            if route and route[0]["actions"]:
                action = route[0]["actions"][0]
                # Override if action is dangerous
                if action not in safe_actions:
                    action = safe_actions[0] if safe_actions else action
                return {
                    "action": action,
                    "source": "plan",
                    "subgoal": current_sg.to_dict(),
                    "plan_status": f"step_{self._current_subgoal_idx}/{len(self._plan)}",
                }

        # No known route — use worker policy with safe action filter
        action = self._worker_action(features, safe_actions)
        return {
            "action": action,
            "source": "worker",
            "subgoal": current_sg.to_dict(),
            "plan_status": f"step_{self._current_subgoal_idx}/{len(self._plan)}",
        }

    def _advance_subgoal(
        self, features: np.ndarray, default_action: Optional[int],
    ) -> Dict[str, Any]:
        """Current subgoal achieved — advance to next."""
        sg = self.current_subgoal
        if sg:
            sg.achieved = True
            self._achieved.add(sg.state_hash)
            self._subgoals_achieved += 1

        self._current_subgoal_idx += 1
        self._steps_on_current = 0

        next_sg = self.current_subgoal
        if next_sg is None:
            self._plans_completed += 1
            return {
                "action": default_action or 0,
                "source": "plan_complete",
                "subgoal": None,
                "plan_status": "complete",
            }

        return {
            "action": default_action or 0,
            "source": "subgoal_achieved",
            "subgoal": next_sg.to_dict(),
            "plan_status": f"step_{self._current_subgoal_idx}/{len(self._plan)}",
        }

    def _handle_subgoal_failure(
        self, features: np.ndarray, default_action: Optional[int],
    ) -> Dict[str, Any]:
        """Subgoal timed out. Replan or abandon."""
        sg = self.current_subgoal
        self._subgoals_failed += 1

        if sg:
            sg.attempts += 1

        if self.replan_on_failure and self._goal_hash is not None:
            current_hash = self.graph._hash_state(features)
            failed_hash = sg.state_hash if sg else 0
            new_plan = self.regressor.replan(self._goal_hash, current_hash, failed_hash)

            if new_plan:
                self._plan = new_plan
                self._current_subgoal_idx = 0
                self._steps_on_current = 0
                return {
                    "action": default_action or 0,
                    "source": "replanned",
                    "subgoal": self.current_subgoal.to_dict() if self.current_subgoal else None,
                    "plan_status": "replanned",
                }

        self._plans_failed += 1
        self._plan = []
        return {
            "action": default_action or 0,
            "source": "plan_failed",
            "subgoal": None,
            "plan_status": "failed",
        }

    def _handle_dead_end(
        self, features: np.ndarray, default_action: Optional[int],
    ) -> Dict[str, Any]:
        """Dead end detected. Mark in graph and replan."""
        self._dead_ends_caught += 1
        current_hash = self.graph._hash_state(features)
        self.graph.mark_dead_end(current_hash)

        return self._handle_subgoal_failure(features, default_action)

    def _check_dead_end(self, features: np.ndarray) -> Dict[str, Any]:
        """Periodic dead-end check during subgoal execution."""
        return {"is_dead_end": self.dead_end_detector.check(features, n_trials=100)}

    def _worker_action(self, features: np.ndarray, safe_actions: List[int]) -> int:
        """Get action from worker (Striatum) filtered by safe actions."""
        try:
            if self.brain.striatum._torch_dqn is not None:
                action, _ = self.brain.striatum._torch_dqn.select_action(
                    features, explore=True,
                )
                if action in safe_actions:
                    return action
                return safe_actions[0] if safe_actions else action
            q_vals = self.brain.striatum._forward(features)
            action = int(np.argmax(q_vals))
            if action in safe_actions:
                return action
            return safe_actions[0] if safe_actions else action
        except Exception:
            return safe_actions[0] if safe_actions else 0

    # ── Observation ──────────────────────────────────────────────────

    def observe_transition(
        self,
        features_before: np.ndarray,
        action: int,
        features_after: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """
        Feed transitions to the causal model and check for traps.

        Called every step to build the causal understanding.
        """
        # Feed causal model
        is_dead = done and reward < 0
        self.causal_model.observe(features_before, action, features_after, reward, is_dead)

        # Check for traps
        if reward > 0 and self.dead_end_detector.is_trap(features_before, action, reward):
            self._traps_caught += 1
            current_hash = self.graph._hash_state(features_after)
            self.graph.mark_trap(current_hash)

        # Update landmark graph with this transition
        from_hash = self.graph.add_landmark(features_before)
        to_hash = self.graph.add_landmark(features_after)

        if done and reward < 0:
            self.graph.record_death(features_after)

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        return {
            "has_plan": self.has_plan,
            "plan_length": len(self._plan),
            "plan_progress": round(self.plan_progress, 2),
            "current_subgoal": self.current_subgoal.to_dict() if self.current_subgoal else None,
            "stats": {
                "plans_made": self._plans_made,
                "plans_completed": self._plans_completed,
                "plans_failed": self._plans_failed,
                "subgoals_achieved": self._subgoals_achieved,
                "subgoals_failed": self._subgoals_failed,
                "dead_ends_caught": self._dead_ends_caught,
                "traps_caught": self._traps_caught,
            },
            "graph": self.graph.report(),
            "causal_model": self.causal_model.report(),
        }
