"""
goal_regression.py — Backward chaining from goal to current state.

Instead of forward search (exponential in puzzles), works backward:
  Goal: "reach exit"
    ← Requires: "have key" (found via landmark graph reverse edges)
      ← Requires: "block dragon" (found via causal model)
        ← Requires: "push rock to row 5"
          ← Requires: "clear path to rock"

Produces a plan as a sequence of subgoals, which the SubgoalPlanner
then executes forward using proven action chains.

Usage:
    regressor = GoalRegression(landmark_graph)
    plan = regressor.regress(goal_hash, current_hash)
    # plan = [subgoal1_hash, subgoal2_hash, ..., goal_hash]
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from brain.planning.landmark_graph import LandmarkGraph


class Subgoal:
    """A single subgoal in a regression-derived plan."""

    def __init__(
        self,
        state_hash: int,
        label: str = "",
        preconditions: Optional[List[int]] = None,
        achieved: bool = False,
        attempts: int = 0,
        max_attempts: int = 10,
    ):
        self.state_hash = state_hash
        self.label = label
        self.preconditions = preconditions or []
        self.achieved = achieved
        self.attempts = attempts
        self.max_attempts = max_attempts

    @property
    def is_exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "label": self.label,
            "preconditions": self.preconditions,
            "achieved": self.achieved,
            "attempts": self.attempts,
        }


class GoalRegression:
    """
    Backward chaining planner.

    Given a goal landmark, works backward through the LandmarkGraph's
    reverse edges to find what preconditions must be met. Produces
    an ordered list of subgoals from current state to goal.

    Also supports causal preconditions: if the CausalModel says
    "action X requires condition Y", Y becomes a subgoal.
    """

    def __init__(
        self,
        graph: LandmarkGraph,
        causal_model=None,
        max_depth: int = 20,
    ):
        self.graph = graph
        self.causal_model = causal_model
        self.max_depth = max_depth

        self._plans_generated: int = 0
        self._last_plan: Optional[List[Subgoal]] = None

    def regress(
        self,
        goal_hash: int,
        current_hash: int,
        avoid_dead_ends: bool = True,
    ) -> Optional[List[Subgoal]]:
        """
        Work backward from goal to current state.

        Returns ordered list of subgoals: [first_to_do, ..., goal].
        Returns None if no path can be found.
        """
        self._plans_generated += 1

        # First try direct graph search (forward Dijkstra)
        goal_lm = self.graph.get_landmark(goal_hash)
        current_lm = self.graph.get_landmark(current_hash)

        if goal_lm is None or current_lm is None:
            return None

        # BFS backward from goal through reverse edges
        visited: Set[int] = set()
        parent: Dict[int, int] = {}
        queue = deque([goal_hash])
        visited.add(goal_hash)

        found = False
        while queue:
            node = queue.popleft()

            if node == current_hash:
                found = True
                break

            # Get all predecessors (reverse edges)
            reverse_edges = self.graph.get_edges_to(node)
            for edge in reverse_edges:
                pred = edge.from_hash
                if pred in visited:
                    continue

                # Skip dead ends
                if avoid_dead_ends:
                    lm = self.graph.get_landmark(pred)
                    if lm and (lm.is_dead_end or lm.is_trap):
                        continue

                visited.add(pred)
                parent[pred] = node
                queue.append(pred)

            # Also check causal preconditions
            if self.causal_model is not None:
                preconditions = self.causal_model.get_preconditions(node)
                for pre_hash in preconditions:
                    if pre_hash not in visited:
                        visited.add(pre_hash)
                        parent[pre_hash] = node
                        queue.append(pre_hash)

        if not found:
            # Try forward search as fallback
            return self._forward_plan(current_hash, goal_hash)

        # Reconstruct forward path: current → ... → goal
        path = []
        node = current_hash
        while node != goal_hash:
            next_node = parent.get(node)
            if next_node is None:
                break

            lm = self.graph.get_landmark(next_node)
            label = lm.label if lm else f"L{next_node}"
            preconditions = []
            if self.causal_model is not None:
                preconditions = self.causal_model.get_preconditions(next_node)

            path.append(Subgoal(
                state_hash=next_node,
                label=label,
                preconditions=preconditions,
            ))
            node = next_node

        self._last_plan = path
        return path

    def _forward_plan(
        self, current_hash: int, goal_hash: int,
    ) -> Optional[List[Subgoal]]:
        """Fallback: use forward Dijkstra from landmark graph."""
        current_lm = self.graph.get_landmark(current_hash)
        goal_lm = self.graph.get_landmark(goal_hash)

        if current_lm is None or goal_lm is None:
            return None
        if current_lm.features is None or goal_lm.features is None:
            return None

        route = self.graph.plan_route(current_lm.features, goal_lm.features)
        if route is None:
            return None

        plan = []
        for step in route:
            lm = self.graph.get_landmark(step["to"])
            label = lm.label if lm else f"L{step['to']}"
            plan.append(Subgoal(
                state_hash=step["to"],
                label=label,
            ))

        self._last_plan = plan
        return plan

    def get_unsatisfied_preconditions(
        self, plan: List[Subgoal], achieved: Set[int],
    ) -> List[Subgoal]:
        """Find subgoals whose preconditions aren't yet met."""
        unsatisfied = []
        for sg in plan:
            for pre_hash in sg.preconditions:
                if pre_hash not in achieved:
                    pre_lm = self.graph.get_landmark(pre_hash)
                    label = pre_lm.label if pre_lm else f"L{pre_hash}"
                    unsatisfied.append(Subgoal(
                        state_hash=pre_hash, label=label,
                    ))
        return unsatisfied

    def replan(
        self, goal_hash: int, current_hash: int, failed_subgoal: int,
    ) -> Optional[List[Subgoal]]:
        """
        Replan after a subgoal fails.

        Marks the failed landmark and tries to find an alternative path.
        """
        # Temporarily mark the failed subgoal's landmark as avoid
        failed_lm = self.graph.get_landmark(failed_subgoal)
        was_dead = False
        if failed_lm:
            was_dead = failed_lm.is_dead_end
            failed_lm.is_dead_end = True

        plan = self.regress(goal_hash, current_hash)

        # Restore
        if failed_lm and not was_dead:
            failed_lm.is_dead_end = False

        return plan

    def report(self) -> Dict[str, Any]:
        return {
            "plans_generated": self._plans_generated,
            "last_plan_length": len(self._last_plan) if self._last_plan else 0,
        }
