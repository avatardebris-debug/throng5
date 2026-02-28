"""
skill_library.py — Parameterized, composable macro-skills.

Skills are higher-level abstractions than action chains:
  - "jump_to(x, y)" — parameterized by target position
  - "collect(item_name)" — parameterized by item
  - "navigate_to(entity)" — uses landmark graph routing
  - "dodge(enemy)" — reactive avoidance pattern

Skills can be composed:
  "reach_key" = navigate_to(key_position) + dodge(dragon) + collect(key)

Each skill has:
  - preconditions (what must be true before starting)
  - effects (what changes after completion)
  - an execution policy (how to achieve it)
  - success/failure tracking

Usage:
    library = SkillLibrary()
    library.register_skill(NavigateSkill("navigate_to"))
    library.register_skill(CollectSkill("collect"))

    composite = library.compose(["navigate_to", "collect"], params={"target": "key"})
    action = composite.get_action(features, game_state)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class Skill(ABC):
    """
    Base class for parameterized skills.

    A skill is a reusable behavior with:
    - preconditions: what must be true before starting
    - effects: what changes after completion
    - parameters: configurable aspects (target position, entity name)
    - execution policy: how to select actions
    """

    def __init__(self, name: str, max_steps: int = 200):
        self.name = name
        self.max_steps = max_steps
        self._params: Dict[str, Any] = {}

        # Execution state
        self._active: bool = False
        self._step_count: int = 0

        # Stats
        self._total_executions: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_reward: float = 0.0

    @property
    def success_rate(self) -> float:
        if self._total_executions == 0:
            return 0.0
        return self._successes / self._total_executions

    def configure(self, **params) -> "Skill":
        """Set parameters for this skill instance. Returns self for chaining."""
        self._params.update(params)
        return self

    def start(self, **params) -> None:
        """Begin skill execution."""
        self._params.update(params)
        self._active = True
        self._step_count = 0
        self._total_executions += 1

    def stop(self, success: bool = False) -> None:
        """End skill execution."""
        self._active = False
        if success:
            self._successes += 1
        else:
            self._failures += 1

    @abstractmethod
    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        """Check if this skill can be started in the current state."""
        ...

    @abstractmethod
    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        """Check if the skill has achieved its goal."""
        ...

    @abstractmethod
    def get_action(
        self, features: np.ndarray, game_state: Dict[str, Any],
    ) -> int:
        """Get the next action for this skill."""
        ...

    def step(
        self, features: np.ndarray, game_state: Dict[str, Any], reward: float,
    ) -> Dict[str, Any]:
        """Execute one step of the skill."""
        self._step_count += 1
        self._total_reward += reward

        # Check completion
        if self.check_completion(game_state):
            self.stop(success=True)
            return {"action": 0, "status": "complete", "skill": self.name}

        # Check timeout
        if self._step_count >= self.max_steps:
            self.stop(success=False)
            return {"action": 0, "status": "timeout", "skill": self.name}

        action = self.get_action(features, game_state)
        return {"action": action, "status": "active", "skill": self.name}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self._params,
            "success_rate": round(self.success_rate, 3),
            "total_executions": self._total_executions,
            "active": self._active,
        }


class NavigateSkill(Skill):
    """Navigate to a target position using proven chains or DQN policy."""

    def __init__(self, name: str = "navigate_to"):
        super().__init__(name, max_steps=300)

    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        return "player_x" in game_state and "target_x" in self._params

    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        if "player_x" not in game_state:
            return False
        tx = self._params.get("target_x", 0)
        ty = self._params.get("target_y", 0)
        px = game_state.get("player_x", 0)
        py = game_state.get("player_y", 0)
        dist = abs(px - tx) + abs(py - ty)
        return dist < self._params.get("threshold", 10)

    def get_action(self, features: np.ndarray, game_state: Dict[str, Any]) -> int:
        tx = self._params.get("target_x", 0)
        ty = self._params.get("target_y", 0)
        px = game_state.get("player_x", 0)
        py = game_state.get("player_y", 0)

        dx = tx - px
        dy = ty - py

        # Simple directional policy (overridden by proven chains if available)
        if abs(dx) > abs(dy):
            return 0 if dx > 0 else 1  # right / left
        else:
            return 2 if dy < 0 else 3  # up / down


class CollectSkill(Skill):
    """Collect an item (navigate to it and pick it up)."""

    def __init__(self, name: str = "collect"):
        super().__init__(name, max_steps=200)

    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        item = self._params.get("item_name", "")
        return item in game_state.get("items", {})

    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        item = self._params.get("item_name", "")
        items = game_state.get("items", {})
        if item in items:
            return items[item].get("collected", False)
        return False

    def get_action(self, features: np.ndarray, game_state: Dict[str, Any]) -> int:
        item = self._params.get("item_name", "")
        items = game_state.get("items", {})
        if item in items:
            ix = items[item].get("x", 0)
            iy = items[item].get("y", 0)
            px = game_state.get("player_x", 0)
            py = game_state.get("player_y", 0)
            dx, dy = ix - px, iy - py
            if abs(dx) > abs(dy):
                return 0 if dx > 0 else 1
            return 2 if dy < 0 else 3
        return np.random.randint(4)


class DodgeSkill(Skill):
    """Dodge an enemy by moving perpendicular to it."""

    def __init__(self, name: str = "dodge"):
        super().__init__(name, max_steps=50)

    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        enemy = self._params.get("enemy_name", "")
        return enemy in game_state.get("enemies", {})

    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        enemy = self._params.get("enemy_name", "")
        enemies = game_state.get("enemies", {})
        if enemy in enemies:
            ex, ey = enemies[enemy].get("x", 0), enemies[enemy].get("y", 0)
            px = game_state.get("player_x", 0)
            py = game_state.get("player_y", 0)
            dist = abs(px - ex) + abs(py - ey)
            return dist > self._params.get("safe_distance", 30)
        return True

    def get_action(self, features: np.ndarray, game_state: Dict[str, Any]) -> int:
        enemy = self._params.get("enemy_name", "")
        enemies = game_state.get("enemies", {})
        if enemy in enemies:
            ex, ey = enemies[enemy].get("x", 0), enemies[enemy].get("y", 0)
            px = game_state.get("player_x", 0)
            py = game_state.get("player_y", 0)
            dx, dy = px - ex, py - ey
            # Move away from enemy
            if abs(dx) > abs(dy):
                return 0 if dx > 0 else 1
            return 2 if dy > 0 else 3
        return np.random.randint(4)


class PushBlockSkill(Skill):
    """Push a block in a specific direction (puzzle-game specific)."""

    def __init__(self, name: str = "push_block"):
        super().__init__(name, max_steps=100)

    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        return "block_x" in self._params

    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        target_x = self._params.get("push_to_x")
        target_y = self._params.get("push_to_y")
        block = self._params.get("block_name", "block")
        blocks = game_state.get("blocks", {})
        if block in blocks:
            bx = blocks[block].get("x", 0)
            by = blocks[block].get("y", 0)
            if target_x is not None and abs(bx - target_x) < 3:
                return True
            if target_y is not None and abs(by - target_y) < 3:
                return True
        return False

    def get_action(self, features: np.ndarray, game_state: Dict[str, Any]) -> int:
        direction = self._params.get("direction", "right")
        # Position on the opposite side of the block, then move toward it
        return {"right": 0, "left": 1, "up": 2, "down": 3}.get(direction, 0)


class CompositeSkill(Skill):
    """A skill composed of sequential sub-skills."""

    def __init__(
        self,
        name: str,
        sub_skills: List[Skill],
    ):
        max_steps = sum(s.max_steps for s in sub_skills)
        super().__init__(name, max_steps=max_steps)
        self._sub_skills = sub_skills
        self._current_idx: int = 0

    @property
    def current_skill(self) -> Optional[Skill]:
        if self._current_idx < len(self._sub_skills):
            return self._sub_skills[self._current_idx]
        return None

    def start(self, **params) -> None:
        super().start(**params)
        self._current_idx = 0
        if self._sub_skills:
            self._sub_skills[0].start(**params)

    def check_preconditions(self, game_state: Dict[str, Any]) -> bool:
        if self._sub_skills:
            return self._sub_skills[0].check_preconditions(game_state)
        return False

    def check_completion(self, game_state: Dict[str, Any]) -> bool:
        return self._current_idx >= len(self._sub_skills)

    def get_action(self, features: np.ndarray, game_state: Dict[str, Any]) -> int:
        current = self.current_skill
        if current is None:
            return 0
        return current.get_action(features, game_state)

    def step(
        self, features: np.ndarray, game_state: Dict[str, Any], reward: float,
    ) -> Dict[str, Any]:
        self._step_count += 1

        current = self.current_skill
        if current is None:
            self.stop(success=True)
            return {"action": 0, "status": "complete", "skill": self.name}

        # Step current sub-skill
        result = current.step(features, game_state, reward)

        if result["status"] == "complete":
            self._current_idx += 1
            next_skill = self.current_skill
            if next_skill:
                next_skill.start(**self._params)
                result["status"] = "advancing"
                result["next_skill"] = next_skill.name
            else:
                self.stop(success=True)
                result["status"] = "complete"

        elif result["status"] in ("timeout", "failed"):
            self.stop(success=False)

        result["composite_progress"] = f"{self._current_idx}/{len(self._sub_skills)}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["sub_skills"] = [s.to_dict() for s in self._sub_skills]
        base["current_idx"] = self._current_idx
        return base


class SkillLibrary:
    """
    Registry and factory for skills.

    Stores skill templates, creates configured instances,
    and composes multi-skill sequences.
    """

    def __init__(self):
        self._templates: Dict[str, Skill] = {}
        self._history: List[Dict[str, Any]] = []

        # Register built-in skills
        self.register_skill(NavigateSkill())
        self.register_skill(CollectSkill())
        self.register_skill(DodgeSkill())
        self.register_skill(PushBlockSkill())

    def register_skill(self, skill: Skill) -> None:
        """Register a skill template."""
        self._templates[skill.name] = skill

    def create(self, name: str, **params) -> Optional[Skill]:
        """Create a configured skill instance."""
        template = self._templates.get(name)
        if template is None:
            return None

        # Create a new instance of the same class
        new_skill = template.__class__(name)
        new_skill.configure(**params)
        return new_skill

    def compose(
        self,
        skill_names: List[str],
        name: str = "",
        **params,
    ) -> Optional[CompositeSkill]:
        """
        Create a composite skill from a sequence of skill names.

        Example:
            compose(["navigate_to", "collect"], name="get_key", target_x=200)
        """
        skills = []
        for sname in skill_names:
            skill = self.create(sname, **params)
            if skill is None:
                return None
            skills.append(skill)

        composite_name = name or "+".join(skill_names)
        return CompositeSkill(composite_name, skills)

    def from_subgoals(
        self, subgoals: List[Dict[str, Any]], game_state: Dict[str, Any],
    ) -> Optional[CompositeSkill]:
        """
        Create a composite skill from LLM-generated subgoals.

        Each subgoal dict should have 'description' and optionally 'target'.
        Maps subgoal descriptions to known skills.
        """
        skills = []
        for sg in subgoals:
            desc = sg.get("description", "").lower()
            target = sg.get("target", "")

            if "navigate" in desc or "move" in desc or "go to" in desc:
                skill = self.create("navigate_to")
                if skill and target:
                    # Try to get target position from game_state
                    items = game_state.get("items", {})
                    if target in items:
                        skill.configure(
                            target_x=items[target].get("x", 0),
                            target_y=items[target].get("y", 0),
                        )
                skills.append(skill)

            elif "collect" in desc or "pick" in desc or "grab" in desc:
                skill = self.create("collect")
                if skill and target:
                    skill.configure(item_name=target)
                skills.append(skill)

            elif "dodge" in desc or "avoid" in desc:
                skill = self.create("dodge")
                if skill and target:
                    skill.configure(enemy_name=target)
                skills.append(skill)

            elif "push" in desc or "block" in desc:
                skill = self.create("push_block")
                if skill:
                    direction = "right"
                    if "left" in desc:
                        direction = "left"
                    elif "up" in desc:
                        direction = "up"
                    elif "down" in desc:
                        direction = "down"
                    skill.configure(direction=direction)
                skills.append(skill)

            else:
                # Default: navigate toward target
                skill = self.create("navigate_to")
                skills.append(skill)

        skills = [s for s in skills if s is not None]
        if not skills:
            return None

        return CompositeSkill("llm_plan", skills)

    def report(self) -> Dict[str, Any]:
        return {
            "registered_skills": list(self._templates.keys()),
            "skill_stats": {
                name: {
                    "success_rate": round(s.success_rate, 3),
                    "executions": s._total_executions,
                }
                for name, s in self._templates.items()
            },
        }
