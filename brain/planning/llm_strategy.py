"""
llm_strategy.py — Format game state for LLM, parse subgoal responses.

The bridge between the ObjectGraph and Tetra (LLM). Formats the
current game state as a structured prompt, sends it to the LLM,
and parses the response into actionable subgoals for the SubgoalPlanner.

The LLM provides:
  - Strategic analysis ("the dragon blocks the key")
  - Subgoal sequences ("1. push rock right, 2. collect key, 3. go to door")
  - Hypothesis generation ("this might be a trap because...")
  - Object identification ("that moving sprite is probably an enemy")

Usage:
    strategy = LLMStrategy(brain, object_graph)
    result = strategy.request_plan(goal="reach the exit")
    subgoals = result["subgoals"]
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from brain.planning.object_graph import ObjectGraph


class LLMStrategy:
    """
    Formats game state for LLM reasoning and parses responses.

    Acts as the translation layer between:
      ObjectGraph (structured data) → LLM prompt (natural language)
      LLM response (natural language) → Subgoal list (structured data)
    """

    def __init__(
        self,
        brain=None,
        object_graph: Optional[ObjectGraph] = None,
    ):
        self.brain = brain
        self.object_graph = object_graph or ObjectGraph()

        self._requests_made: int = 0
        self._last_response: Optional[str] = None
        self._history: List[Dict[str, Any]] = []

    # ── Prompt Building ──────────────────────────────────────────────

    def build_state_prompt(
        self,
        goal: str = "",
        context: str = "",
        stuck_points: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build a structured prompt describing the current game state.

        Includes entity descriptions, relationships, recent history,
        and the specific question/goal.
        """
        sections = []

        # Game state from ObjectGraph
        state_desc = self.object_graph.describe()
        if state_desc:
            sections.append(f"# Current Game State\n{state_desc}")

        # Stuck points
        if stuck_points:
            sections.append("# Known Stuck Points (repeated failures)")
            for sp in stuck_points[:5]:
                sections.append(f"- State #{sp.get('state_hash', '?')}: {sp.get('death_count', 0)} deaths")

        # Context
        if context:
            sections.append(f"# Context\n{context}")

        # Goal
        if goal:
            sections.append(f"# Goal\n{goal}")

        # Request
        sections.append("""# Request
Based on the game state above:
1. What objects and relationships do you observe?
2. What is blocking progress toward the goal?
3. What sequence of subgoals would achieve the goal?
4. Are there any potential traps (actions that give reward but block progress)?

Format subgoals as a numbered list:
1. [SUBGOAL] description (target: entity_name or position)
2. [SUBGOAL] description
...""")

        return "\n\n".join(sections)

    def build_hypothesis_prompt(
        self,
        observation: str,
        ram_changes: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build a prompt asking the LLM to hypothesize about game mechanics.

        Used when the agent encounters unknown objects or effects.
        """
        sections = [f"# Observation\n{observation}"]

        if ram_changes:
            sections.append("# RAM Changes Observed")
            for change in ram_changes[:10]:
                sections.append(
                    f"- Address {change.get('addr_hex', '??')}: "
                    f"{change.get('old', '?')} → {change.get('new', '?')}"
                )

        state_desc = self.object_graph.describe()
        if state_desc:
            sections.append(f"# Current State\n{state_desc}")

        sections.append("""# Request
Based on these observations:
1. What game mechanic might explain these changes?
2. What entity/object do you think changed?
3. Is this change beneficial, harmful, or neutral?
4. What should the agent do differently based on this?

Format as:
HYPOTHESIS: [your hypothesis]
ENTITY: [entity name]
EFFECT: [beneficial/harmful/neutral]
ACTION: [suggested response]""")

        return "\n\n".join(sections)

    def build_object_identification_prompt(
        self,
        ram_registry: Dict[str, List],
        entity_groups: List[Dict],
    ) -> str:
        """
        Ask the LLM to name discovered RAM entities.

        Provides the statistical profile and asks for human-readable names.
        """
        sections = ["# Discovered RAM Entities\n"]

        for group in entity_groups:
            sections.append(
                f"- Entity '{group['type']}': RAM bytes {group['bytes']}, "
                f"labels: {group['labels']}"
            )

        if "state_flag" in ram_registry:
            sections.append("\n# State Flags (binary changes)")
            for item in ram_registry["state_flag"][:10]:
                sections.append(
                    f"- {item['addr_hex']}: {item['label']} "
                    f"(reward-correlated: {item.get('reward_correlated', False)})"
                )

        sections.append("""
# Request
Based on typical NES/Atari game mechanics, what are these entities likely to be?
For each entity, provide:
NAME: [human-readable name, e.g. "player", "key", "enemy_knight"]
TYPE: [player/item/enemy/obstacle/goal]
NOTES: [any relevant observations]""")

        return "\n\n".join(sections)

    # ── Response Parsing ─────────────────────────────────────────────

    def parse_subgoals(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response into structured subgoals.

        Looks for numbered lists with [SUBGOAL] markers or plain numbered items.
        """
        subgoals = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Match: "1. [SUBGOAL] description" or "1. description"
            match = re.match(
                r"^(\d+)\.\s*(?:\[SUBGOAL\]\s*)?(.+?)(?:\(target:\s*(.+?)\))?$",
                line, re.IGNORECASE,
            )
            if match:
                idx = int(match.group(1))
                description = match.group(2).strip()
                target = match.group(3).strip() if match.group(3) else None

                subgoals.append({
                    "index": idx,
                    "description": description,
                    "target": target,
                    "source": "llm",
                })

        return subgoals

    def parse_hypothesis(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis response into structured data."""
        result = {
            "hypothesis": "",
            "entity": "",
            "effect": "neutral",
            "action": "",
            "raw": response,
        }

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("HYPOTHESIS:"):
                result["hypothesis"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ENTITY:"):
                result["entity"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EFFECT:"):
                result["effect"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("ACTION:"):
                result["action"] = line.split(":", 1)[1].strip()

        return result

    def parse_entity_names(self, response: str) -> List[Dict[str, str]]:
        """Parse entity naming response."""
        entities = []
        current: Dict[str, str] = {}

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("NAME:"):
                if current:
                    entities.append(current)
                current = {"name": line.split(":", 1)[1].strip()}
            elif line.upper().startswith("TYPE:"):
                current["type"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("NOTES:"):
                current["notes"] = line.split(":", 1)[1].strip()

        if current:
            entities.append(current)
        return entities

    # ── High-Level API ───────────────────────────────────────────────

    def request_plan(
        self,
        goal: str,
        context: str = "",
        stuck_points: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Request a strategic plan from the LLM.

        Builds prompt, queries LLM (or returns rule-based fallback),
        and parses the response into subgoals.
        """
        self._requests_made += 1

        prompt = self.build_state_prompt(goal, context, stuck_points)

        # Try LLM via Prefrontal Cortex
        response = self._query_llm(prompt)
        subgoals = self.parse_subgoals(response)

        result = {
            "prompt": prompt,
            "response": response,
            "subgoals": subgoals,
            "n_subgoals": len(subgoals),
            "source": "llm" if subgoals else "fallback",
        }

        self._last_response = response
        self._history.append({
            "time": time.time(),
            "goal": goal,
            "n_subgoals": len(subgoals),
        })

        return result

    def request_hypothesis(
        self,
        observation: str,
        ram_changes: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Request LLM hypothesis about observed game mechanics."""
        self._requests_made += 1
        prompt = self.build_hypothesis_prompt(observation, ram_changes)
        response = self._query_llm(prompt)
        return self.parse_hypothesis(response)

    def request_entity_names(
        self,
        ram_registry: Dict[str, List],
        entity_groups: List[Dict],
    ) -> List[Dict[str, str]]:
        """Ask LLM to name discovered entities."""
        self._requests_made += 1
        prompt = self.build_object_identification_prompt(ram_registry, entity_groups)
        response = self._query_llm(prompt)
        return self.parse_entity_names(response)

    def _query_llm(self, prompt: str) -> str:
        """
        Send prompt to LLM via Prefrontal Cortex bridge.

        Falls back to rule-based response if LLM is unavailable.
        """
        # Try Tetra/LLM bridge via Prefrontal Cortex
        if self.brain is not None and hasattr(self.brain, 'prefrontal'):
            try:
                result = self.brain.prefrontal._publish(
                    "llm_strategy_request",
                    {"prompt": prompt},
                )
                if isinstance(result, str) and len(result) > 10:
                    return result
            except Exception:
                pass

        # Rule-based fallback: extract entities and build simple plan
        return self._rule_based_fallback(prompt)

    def _rule_based_fallback(self, prompt: str) -> str:
        """
        Generate a basic plan from the ObjectGraph when LLM is unavailable.

        Uses simple heuristics:
        1. If there's a goal entity, plan a path to it
        2. If something blocks the goal, plan to remove the blocker first
        3. If items are required, collect them first
        """
        lines = []
        step = 1

        graph = self.object_graph
        goals = graph.get_entities_by_category("goal")
        items = graph.get_entities_by_category("item")
        player = graph.get_entities_by_category("player")

        # Collect uncollected items
        for item in items:
            if not item.properties.get("collected", True):
                # Check if blocked
                blockers = graph.get_blockers(item.name)
                for blocker in blockers:
                    lines.append(f"{step}. [SUBGOAL] Remove blocker {blocker} (target: {blocker})")
                    step += 1
                lines.append(f"{step}. [SUBGOAL] Collect {item.name} (target: {item.name})")
                step += 1

        # Go to goal
        for goal in goals:
            requirements = graph.get_requirements(goal.name)
            for req in requirements:
                lines.append(f"{step}. [SUBGOAL] Ensure {req} is satisfied (target: {req})")
                step += 1
            lines.append(f"{step}. [SUBGOAL] Reach {goal.name} (target: {goal.name})")
            step += 1

        if not lines:
            lines.append("1. [SUBGOAL] Explore to find objectives (target: unknown)")

        return "\n".join(lines)

    def report(self) -> Dict[str, Any]:
        return {
            "requests_made": self._requests_made,
            "history_length": len(self._history),
            "entities_in_graph": len(self.object_graph._entities),
            "relations_in_graph": len(self.object_graph._relations),
        }
