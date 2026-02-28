"""
run_montezuma_brain.py — Run Montezuma's Revenge with the full WholeBrain stack.

MODES:
  human     — YOU play with keyboard, game renders, RAM + actions recorded
  watch     — Agent plays with rendering so you can observe behavior
  ground    — Analyze a human recording to discover RAM semantics
  train     — Standard training loop (no rendering, max speed)
  plan      — Goal-directed training with subgoal planner active
  rehearse  — Focused bottleneck practice with save states

Usage:
    python run_montezuma_brain.py human                 # Play yourself, record RAM
    python run_montezuma_brain.py watch --episodes 5    # Watch agent play
    python run_montezuma_brain.py ground recording.jsonl # Analyze human recording
    python run_montezuma_brain.py train --episodes 500  # Train (no render)
    python run_montezuma_brain.py plan  --episodes 200  # Goal-directed training
    python run_montezuma_brain.py rehearse              # Practice bottlenecks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ALE/Gymnasium
import warnings
import ale_py
import gymnasium as gym
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ale_py.register_v5_envs()

# Brain
from brain.orchestrator import WholeBrain

# Planning layer
from brain.planning.ram_semantic_mapper import RAMSemanticMapper
from brain.planning.reward_discovery import RewardDiscovery
from brain.planning.object_graph import ObjectGraph
from brain.planning.meta_planner import MetaPlanner
from brain.planning.safety import SafetyConstraints
from brain.planning.temporal import TemporalReasoner
from brain.planning.skill_library import SkillLibrary
from brain.planning.procedural_memory import ProceduralMemory
from brain.planning.self_model import SelfModel

# Human recording
from brain.environments.human_recorder import HumanRecorder


# ── Constants ────────────────────────────────────────────────────────

GAME_ID = "ALE/MontezumaRevenge-v5"
N_ACTIONS = 18
RAM_SIZE = 128
RESULTS_DIR = Path("experiments/montezuma_brain")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR = RESULTS_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Montezuma RAM Addresses ─────────────────────────────────────────

RAM_PLAYER_X    = 42
RAM_PLAYER_Y    = 43
RAM_ROOM        = 3
RAM_LIVES       = 58
RAM_SCORE_1     = 19
RAM_SCORE_2     = 20
RAM_SKULL_X     = 47
RAM_SKULL_Y     = 46
RAM_ITEMS       = 65

# Atari action names for display
ACTION_NAMES = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]

# Keyboard mapping for human play (pygame key → atari action)
# Standard: arrow keys + space(fire), diagonals with shift
KEYBOARD_MAP = {
    "noop":       0,
    "fire":       1,
    "up":         2,
    "right":      3,
    "left":       4,
    "down":       5,
    "upright":    6,
    "upleft":     7,
    "downright":  8,
    "downleft":   9,
    "upfire":    10,
    "rightfire": 11,
    "leftfire":  12,
    "downfire":  13,
}


# ── Exploration Manager ─────────────────────────────────────────────

class ExplorationManager:
    """
    Prevents the agent from getting stuck via:
    1. High initial epsilon that decays over episodes (1.0 → 0.05)
    2. Anti-stuck: if position unchanged for N steps, force random action
    3. Position novelty: bonus for visiting new (x,y,room) combos
    4. NOOP suppression: Montezuma rarely benefits from NOOP
    """

    def __init__(self, n_actions: int = 18, decay_episodes: int = 300):
        self._n_actions = n_actions
        self._decay_episodes = decay_episodes
        self._episode = 0

        # Epsilon schedule
        self._epsilon_start = 1.0
        self._epsilon_end = 0.05
        self._epsilon = self._epsilon_start

        # Anti-stuck tracking
        self._last_pos = (-1, -1, -1)  # (x, y, room)
        self._stuck_steps = 0
        self._stuck_threshold = 30  # Force random after this many identical frames

        # Position novelty
        self._visited: set = set()
        self._visit_counts: dict = {}

        # Stats
        self._forced_random = 0
        self._total_selects = 0

    def on_episode_start(self):
        """Call at episode start to update epsilon."""
        self._episode += 1
        # Linear decay
        frac = min(1.0, self._episode / self._decay_episodes)
        self._epsilon = self._epsilon_start + frac * (self._epsilon_end - self._epsilon_start)
        self._stuck_steps = 0
        self._last_pos = (-1, -1, -1)

    def select_action(self, brain_action: int, game_state: dict, features: np.ndarray) -> int:
        """
        Filter or override the brain's action.

        Returns the final action to take.
        """
        self._total_selects += 1
        pos = (game_state["player_x"], game_state["player_y"], game_state["room"])

        # Anti-stuck: if position hasn't changed, increment counter
        if pos == self._last_pos:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
            self._last_pos = pos

        # Force random action if stuck (position unchanged for too long)
        if self._stuck_steps > self._stuck_threshold:
            self._forced_random += 1
            # Bias toward movement actions (UP=2, RIGHT=3, LEFT=4, DOWN=5, combos=6-9)
            action = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            self._stuck_steps = 0
            return action

        # Epsilon-greedy override (higher epsilon than brain's default)
        if np.random.random() < self._epsilon:
            # Weighted random: prefer movement actions over NOOP
            weights = np.ones(self._n_actions)
            weights[0] = 0.02  # Suppress NOOP
            weights[1:6] = 2.0  # Boost basic moves
            weights[6:10] = 1.5  # Boost diagonal moves
            weights[10:] = 1.0  # Fire combos
            weights /= weights.sum()
            return int(np.random.choice(self._n_actions, p=weights))

        return brain_action

    def get_novelty_bonus(self, game_state: dict) -> float:
        """Position-based novelty reward: bonus for new (x,y,room) combos."""
        # Quantize position to 8x8 grid cells
        cell = (
            game_state["player_x"] // 8,
            game_state["player_y"] // 8,
            game_state["room"],
        )

        key = cell
        self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
        count = self._visit_counts[key]

        if key not in self._visited:
            self._visited.add(key)
            return 1.0  # Big bonus for first visit

        # Decaying bonus: 1/sqrt(n)
        return 0.1 / np.sqrt(count)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def report(self) -> dict:
        return {
            "epsilon": round(self._epsilon, 3),
            "episode": self._episode,
            "unique_positions": len(self._visited),
            "forced_random": self._forced_random,
            "total_selects": self._total_selects,
        }


# ── Fall Predictor (Predictive, not Reactive) ───────────────────────

class FallPredictor:
    """
    PREDICTIVE fall avoidance — filters dangerous actions BEFORE taking them.

    Instead of detecting falls mid-air (too late to recover), this:
    1. Learns which (position, action) combos led to falls
    2. Knows Montezuma platform geometry (hardcoded boot knowledge)
    3. Before action selection, removes actions likely to cause falls

    Rope zone (x=67-145) is exempt — y-drops there are expected.
    """

    FALL_THRESHOLD = 20  # y-drop bigger than a jump = fall
    ROPE_X_MIN = 67
    ROPE_X_MAX = 145
    BIN_SIZE = 8  # Spatial binning for position lookup
    LEARN_THRESHOLD = 2  # Falls before banning an action at a position

    # Atari Montezuma action mapping
    # 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN
    # 6=UPRIGHT, 7=UPLEFT, 8=DOWNRIGHT, 9=DOWNLEFT
    # 10=UPFIRE, 11=RIGHTFIRE, 12=LEFTFIRE, 13=DOWNFIRE
    # 14=UPRIGHTFIRE, 15=UPLEFTFIRE, 16=DOWNRIGHTFIRE, 17=DOWNLEFTFIRE
    RIGHTWARD_ACTIONS = {3, 6, 8, 11, 14, 16}   # Actions that move right
    LEFTWARD_ACTIONS = {4, 7, 9, 12, 15, 17}     # Actions that move left
    DOWNWARD_ACTIONS = {5, 8, 9, 13, 16, 17}      # Actions that move down

    def __init__(self):
        # Learned danger map: (x_bin, y_bin, room) → {action: fall_count}
        self._danger_map = {}

        # Position history for fall detection (to learn FROM)
        self._pos_history = []  # [(x, y, room, action)]
        self._prev_y = None
        self._prev_x = None
        self._prev_room = None
        self._prev_action = None

        # Hardcoded Montezuma room 0 platform edges
        # These are x-values where the platform ends
        # Going further in that direction = falling
        self._platform_edges = {
            # room: [(x_left_edge, x_right_edge, y_level, tolerance)]
            0: [
                # Top platform (spawn level): x ≈ 5-149, y ≈ 148
                (5, 149, 148, 4),
                # Middle-left platform: x ≈ 5-60, y ≈ 185
                (5, 60, 185, 4),
                # Middle-right platform: x ≈ 100-149, y ≈ 185
                (100, 149, 185, 4),
                # Bottom platform: x ≈ 5-149, y ≈ 235
                (5, 149, 235, 4),
            ],
            1: [
                (5, 149, 148, 4),
                (5, 60, 185, 4),
                (100, 149, 185, 4),
                (5, 149, 235, 4),
            ],
        }

        # Stats
        self._falls_learned = 0
        self._actions_blocked = 0
        self._edge_warnings = 0

    def on_episode_start(self):
        """Reset per-episode tracking (learned knowledge persists)."""
        self._pos_history = []
        self._prev_y = None
        self._prev_x = None
        self._prev_room = None
        self._prev_action = None

    def _bin(self, x, y):
        """Bin position for spatial lookup."""
        return (x // self.BIN_SIZE, y // self.BIN_SIZE)

    def observe(self, game_state: dict, action_taken: int):
        """
        Call AFTER stepping the environment. Detects if a fall happened
        and records the (position, action) that caused it for future avoidance.
        """
        px = game_state["player_x"]
        py = game_state["player_y"]
        room = game_state["room"]
        in_rope_zone = self.ROPE_X_MIN <= px <= self.ROPE_X_MAX

        if self._prev_y is not None and not in_rope_zone:
            y_delta = py - self._prev_y  # Positive = falling down

            if y_delta > self.FALL_THRESHOLD:
                # A fall happened! Record the PREVIOUS position + action as dangerous
                bx, by = self._bin(self._prev_x, self._prev_y)
                key = (bx, by, self._prev_room)

                if key not in self._danger_map:
                    self._danger_map[key] = {}

                prev_act = self._prev_action
                self._danger_map[key][prev_act] = self._danger_map[key].get(prev_act, 0) + 1
                self._falls_learned += 1

                # Also learn nearby positions (the approach was dangerous too)
                for hist_x, hist_y, hist_room, hist_act in self._pos_history[-5:]:
                    hkey = (hist_x // self.BIN_SIZE, hist_y // self.BIN_SIZE, hist_room)
                    if hkey not in self._danger_map:
                        self._danger_map[hkey] = {}
                    self._danger_map[hkey][hist_act] = self._danger_map[hkey].get(hist_act, 0) + 1

        # Update history
        self._pos_history.append((px, py, room, action_taken))
        if len(self._pos_history) > 20:
            self._pos_history.pop(0)

        self._prev_x = px
        self._prev_y = py
        self._prev_room = room
        self._prev_action = action_taken

    def filter_actions(self, game_state: dict, available_actions: list) -> list:
        """
        BEFORE action selection: remove actions predicted to cause falls.

        Uses:
        1. Learned danger map (prior falls at this position)
        2. Platform edge geometry (hardcoded Montezuma knowledge)
        3. Rope zone exemption

        Returns filtered action list (always at least 1 action).
        """
        px = game_state["player_x"]
        py = game_state["player_y"]
        room = game_state["room"]

        # Exempt rope zone
        if self.ROPE_X_MIN <= px <= self.ROPE_X_MAX:
            return available_actions

        dangerous_actions = set()

        # ── 1. Learned danger map ─────────────────────────────────
        bx, by = self._bin(px, py)
        key = (bx, by, room)
        if key in self._danger_map:
            for act, count in self._danger_map[key].items():
                if count >= self.LEARN_THRESHOLD:
                    dangerous_actions.add(act)

        # Also check adjacent bins (danger zone is fuzzy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                adj_key = (bx + dx, by + dy, room)
                if adj_key in self._danger_map:
                    for act, count in self._danger_map[adj_key].items():
                        if count >= self.LEARN_THRESHOLD + 1:  # Higher bar for neighbors
                            dangerous_actions.add(act)

        # ── 2. Platform edge geometry ─────────────────────────────
        edges = self._platform_edges.get(room, [])
        for x_left, x_right, y_level, tol in edges:
            if abs(py - y_level) <= tol:
                # On this platform
                edge_margin = 12  # Pixels from edge to start being cautious

                # Near left edge → don't go further left
                if px <= x_left + edge_margin:
                    dangerous_actions.update(self.LEFTWARD_ACTIONS)
                    self._edge_warnings += 1

                # Near right edge → don't go further right
                if px >= x_right - edge_margin:
                    dangerous_actions.update(self.RIGHTWARD_ACTIONS)
                    self._edge_warnings += 1

        # ── 3. Filter ────────────────────────────────────────────
        safe = [a for a in available_actions if a not in dangerous_actions]

        if dangerous_actions:
            self._actions_blocked += len(dangerous_actions)

        # Always return at least one action
        if not safe:
            # Prefer UP/NOOP as safest fallbacks
            for fallback in [2, 0, 1]:  # UP, NOOP, FIRE
                if fallback in available_actions:
                    return [fallback]
            return available_actions[:1]

        return safe

    def report(self) -> dict:
        return {
            "falls_learned": self._falls_learned,
            "danger_zones": len(self._danger_map),
            "actions_blocked": self._actions_blocked,
            "edge_warnings": self._edge_warnings,
        }



from brain.planning.landmark_graph import LandmarkGraph
from brain.planning.goal_regression import GoalRegression
from brain.planning.dead_end_detector import DeadEndDetector
from brain.planning.causal_model import CausalModel
from brain.planning.subgoal_planner import SubgoalPlanner
from brain.planning.llm_strategy import LLMStrategy
from brain.planning.counterfactual import CounterfactualReasoner


class IntentionalController:
    """
    Closed-loop planning controller.

    Orchestrates: Grounding → Subgoals → Skills → Reward Shaping → Action.

    Flow per step:
      1. Update landmark graph (room change → new landmark)
      2. Generate subgoals if none active (LLMStrategy → SubgoalPlanner)
      3. Shape reward toward current subgoal (distance-based)
      4. Pick action: skill action or DQN + counterfactual
      5. Learn causal effects from transition
    """

    def __init__(self, brain, object_graph, n_actions: int = 18):
        self.brain = brain
        self.object_graph = object_graph
        self.n_actions = n_actions

        # Planning stack
        self.landmark_graph = LandmarkGraph()
        self.causal_model = CausalModel()
        self.dead_end_detector = DeadEndDetector(brain, default_trials=50, rollout_length=30)
        self.goal_regression = GoalRegression(self.landmark_graph, self.causal_model)
        self.subgoal_planner = SubgoalPlanner(
            brain, self.landmark_graph, self.goal_regression,
            self.dead_end_detector, self.causal_model,
            subgoal_timeout=300,
        )

        # Strategy + Skills
        self.llm_strategy = LLMStrategy(brain, object_graph=object_graph)
        self.skill_lib = SkillLibrary()
        self.counterfactual = CounterfactualReasoner(brain)

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
                landmark_hash = self.landmark_graph.add_landmark(
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
            if game_state["items"] != self._prev_game_state.get("items", 0):
                self.landmark_graph.add_landmark(
                    features, label=f"item_{game_state['items']:08b}",
                    is_goal=True, step=self._steps_since_plan,
                )
                self._landmarks_found += 1
                shaped_reward += 5.0  # Big bonus for item collection

            # Death tracking
            if done and reward <= 0:
                self.landmark_graph.record_death(features)

        # ─── 2. Causal model: learn effects ───────────────────────
        if self._prev_features is not None:
            self.causal_model.observe(
                self._prev_features, brain_action, features,
                reward=reward, is_dead_end=(done and reward <= 0),
            )
            self._causal_observations += 1

            # Feed SubgoalPlanner transition observer
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
            # Compare top-3 Q-value actions
            try:
                q_vals = self.brain.basal_ganglia.process(
                    {"features": features}
                ).get("q_values", None)
                if q_vals is not None:
                    top_actions = np.argsort(q_vals)[-3:].tolist()
                    comparison = self.counterfactual.compare_actions(
                        features, top_actions, n_steps=20,
                    )
                    cf_best = comparison.get("best_action", brain_action)
                    if comparison.get("regret", 0) > 0.5:
                        action = cf_best
                        info["source"] = "counterfactual"
            except Exception:
                pass  # Counterfactual is optional

        # Update state
        self._prev_game_state = game_state.copy()
        self._prev_features = features.copy()

        return {
            "action": action,
            "shaped_reward": shaped_reward,
            "info": info,
        }

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
                    f"items={game_state['items']:08b}, "
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


def get_game_state(ram: np.ndarray) -> dict:
    """Extract semantic game state from RAM."""
    return {
        "player_x": int(ram[RAM_PLAYER_X]),
        "player_y": int(ram[RAM_PLAYER_Y]),
        "room": int(ram[RAM_ROOM]),
        "lives": int(ram[RAM_LIVES]),
        "score": int(ram[RAM_SCORE_1]) * 100 + int(ram[RAM_SCORE_2]),
        "skull_x": int(ram[RAM_SKULL_X]),
        "skull_y": int(ram[RAM_SKULL_Y]),
        "items": int(ram[RAM_ITEMS]),
        "has_key": bool(ram[RAM_ITEMS] & 0x01),
    }


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
        "skill_lib": SkillLibrary(),
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


# ═══════════════════════════════════════════════════════════════════
# MODE 1: HUMAN PLAY — You play, RAM is recorded for grounding
# ═══════════════════════════════════════════════════════════════════

def mode_human(args):
    """
    Human play mode.

    Renders the game window. You play with keyboard (via pygame).
    Every frame, RAM + action + reward are recorded to JSONL.
    After you quit, the recording is analyzed to discover RAM semantics.

    Controls:
      Arrow keys = move (UP/DOWN/LEFT/RIGHT)
      Space      = FIRE (jump)
      Q          = quit and save recording
    """
    print("=" * 60)
    print("MODE: HUMAN PLAY")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Arrow keys = move")
    print("  Space      = jump/fire")
    print("  Q          = quit and save")
    print()

    # Try importing pygame for keyboard input
    try:
        import pygame
        has_pygame = True
    except ImportError:
        has_pygame = False
        print("WARNING: pygame not installed. Using random actions.")
        print("Install with: pip install pygame")
        print()

    env = make_env(render=True, frameskip=args.frameskip)
    recorder = HumanRecorder("human_play")

    session_name = f"human_{int(time.time())}"
    total_frames = 0
    total_reward = 0.0
    best_room = 0
    episode = 0

    if has_pygame:
        pygame.init()
        # We need a tiny window to capture keyboard events
        # (the ALE render window doesn't capture keys)
        key_surface = pygame.display.set_mode((200, 100))
        pygame.display.set_caption("Montezuma Keys - Press Q to quit")
        font = pygame.font.SysFont("monospace", 14)

    try:
        for ep in range(args.episodes):
            episode = ep
            obs, info = env.reset()
            ram = env.unwrapped.ale.getRAM()
            recorder.start(None)

            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done and ep_steps < args.max_steps:
                # Get keyboard action
                action = 0  # NOOP
                quit_requested = False

                if has_pygame:
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_q]:
                        quit_requested = True
                    elif keys[pygame.K_SPACE] and keys[pygame.K_UP]:
                        action = 10   # UPFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
                        action = 11   # RIGHTFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
                        action = 12   # LEFTFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_DOWN]:
                        action = 13   # DOWNFIRE
                    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                        action = 6    # UPRIGHT
                    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                        action = 7    # UPLEFT
                    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                        action = 8    # DOWNRIGHT
                    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                        action = 9    # DOWNLEFT
                    elif keys[pygame.K_SPACE]:
                        action = 1    # FIRE
                    elif keys[pygame.K_UP]:
                        action = 2    # UP
                    elif keys[pygame.K_RIGHT]:
                        action = 3    # RIGHT
                    elif keys[pygame.K_LEFT]:
                        action = 4    # LEFT
                    elif keys[pygame.K_DOWN]:
                        action = 5    # DOWN

                    # Update key window
                    game_state = get_game_state(ram)
                    key_surface.fill((0, 0, 0))
                    lines = [
                        f"Rm:{game_state['room']} "
                        f"({game_state['player_x']},{game_state['player_y']})",
                        f"R={ep_reward:.0f} "
                        f"Act={ACTION_NAMES[action]:10s}",
                        f"Lives={game_state['lives']} "
                        f"Items={game_state['items']:08b}",
                    ]
                    for i, line in enumerate(lines):
                        surf = font.render(line, True, (0, 255, 0))
                        key_surface.blit(surf, (5, 5 + i * 20))
                    pygame.display.flip()

                    if quit_requested:
                        print("\nQuit requested. Saving recording...")
                        break
                else:
                    # No pygame — use simple random actions for testing
                    action = np.random.randint(N_ACTIONS)

                # Record
                ram = env.unwrapped.ale.getRAM()
                recorder.record(ram, action, ep_reward, done)

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1
                total_frames += 1
                ram = env.unwrapped.ale.getRAM()

                game_state = get_game_state(ram)
                best_room = max(best_room, game_state["room"])

                # Slow down to human speed
                time.sleep(0.016)  # ~60fps

            recorder.stop()
            total_reward += ep_reward
            print(f"  Episode {ep}: reward={ep_reward:.0f}, "
                  f"room={game_state['room']}, steps={ep_steps}")

            if quit_requested:
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving recording...")

    # Save recording
    rec_path = recorder.save(str(RECORDINGS_DIR))
    print(f"\nRecording saved to {rec_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Best room: {best_room}")

    # Analyze
    analysis = recorder.analyze()
    print(f"\nRAM Analysis:")
    print(f"  Position candidates: {len(analysis.get('position_candidates', []))}")
    print(f"  State flags: {len(analysis.get('state_flag_candidates', []))}")
    print(f"  Reward events: {analysis.get('reward_events', 0)}")

    # Save analysis
    analysis_path = RECORDINGS_DIR / f"{session_name}_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis saved to {analysis_path}")

    if has_pygame:
        pygame.quit()
    env.close()


# ═══════════════════════════════════════════════════════════════════
# MODE 2: WATCH — Agent plays with rendering
# ═══════════════════════════════════════════════════════════════════

def mode_watch(args):
    """
    Watch the agent play with rendering.

    Now with full intentional planning: subgoals, skills, reward shaping,
    and counterfactual action comparison.
    """
    print("=" * 60)
    print("MODE: WATCH AGENT PLAY (Intentional)")
    print("=" * 60)

    env = make_env(render=True, frameskip=args.frameskip)
    brain = WholeBrain(n_features=RAM_SIZE, n_actions=N_ACTIONS,
                       session_name="montezuma_watch", use_cnn=False)
    modules = make_modules()
    meta_planner = MetaPlanner(brain)
    self_model = SelfModel(brain)
    explorer = ExplorationManager(n_actions=N_ACTIONS, decay_episodes=max(args.episodes // 2, 10))
    fall_predictor = FallPredictor()
    controller = IntentionalController(brain, modules["object_graph"], n_actions=N_ACTIONS)

    # Start in planning mode immediately
    controller.set_mode("planning")

    for episode in range(args.episodes):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()
        prev_ram = ram.copy()
        explorer.on_episode_start()
        fall_predictor.on_episode_start()
        game_state = get_game_state(ram)
        controller.on_episode_start(game_state)

        ep_reward = 0.0
        ep_steps = 0
        prev_action = 0
        done = False
        last_subgoal = None

        while not done and ep_steps < args.max_steps:
            ram = env.unwrapped.ale.getRAM()
            game_state = get_game_state(ram)

            modules["ram_mapper"].observe(ram, action=prev_action, reward=ep_reward, done=done)
            modules["temporal"].observe(ram, step=ep_steps)

            intrinsic_r = modules["reward_disc"].compute(
                prev_ram, ram, action=prev_action, done=done,
            )
            # Add position novelty bonus
            intrinsic_r += explorer.get_novelty_bonus(game_state)

            features = ram.astype(np.float32) / 255.0
            result = brain.step(features, prev_action=prev_action, reward=intrinsic_r, done=False)
            brain_action = result["action"] if isinstance(result, dict) else result

            # ── Intentional Controller: subgoals + skills + shaping ──
            ctrl_result = controller.step(
                game_state, features, brain_action,
                reward=ep_reward, done=done,
            )
            action = ctrl_result["action"]
            intrinsic_r += ctrl_result["shaped_reward"]
            ctrl_info = ctrl_result["info"]

            # Log subgoal changes
            if ctrl_info.get("subgoal") != last_subgoal and ctrl_info.get("subgoal"):
                print(f"  🎯 Subgoal: {ctrl_info['subgoal']} (via {ctrl_info['source']})")
                last_subgoal = ctrl_info.get("subgoal")

            # Predictive fall avoidance: filter actions BEFORE selection
            fall_safe = fall_predictor.filter_actions(game_state, list(range(N_ACTIONS)))

            # Exploration override (safety net)
            action = explorer.select_action(action, game_state, features)

            safe_actions = modules["safety"].filter_actions(
                features, fall_safe,
                player_pos=(game_state["player_x"], game_state["player_y"]),
            )
            if action not in safe_actions:
                action = safe_actions[0] if safe_actions else action

            meta_planner.observe(features, reward=intrinsic_r, done=False, action=action)
            modules["proc_memory"].observe_transition(action, features, features, intrinsic_r)

            # Update object graph
            modules["object_graph"].add_entity("player", {
                "x": game_state["player_x"], "y": game_state["player_y"],
            }, category="player")
            modules["object_graph"].add_entity("skull", {
                "x": game_state["skull_x"], "y": game_state["skull_y"],
            }, category="enemy")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            prev_action = action
            prev_ram = ram.copy()

            # Learn from falls (after step, so we see the y-delta)
            fall_predictor.observe(get_game_state(env.unwrapped.ale.getRAM()), action)

            if done and reward <= 0:
                modules["safety"].learn_from_death(features, prev_action, context="death")
                self_model.record_death("death", features)

            self_model.record_position(
                game_state["player_x"], game_state["player_y"],
                area_hash=game_state["room"],
            )

        brain.step(ram.astype(np.float32) / 255.0,
                   prev_action=prev_action, reward=ep_reward, done=True)

        mode = meta_planner.decide()
        # Sync MetaPlanner mode → controller mode
        if mode in ("planning", "explore"):
            controller.set_mode(mode)
        elif mode == "reactive":
            controller.set_mode("reactive")

        ctrl_rpt = controller.report()
        print(f"Ep {episode:3d} | R={ep_reward:6.0f} | "
              f"room={game_state['room']} | steps={ep_steps} | "
              f"mode={mode} | ε={explorer.epsilon:.2f} | "
              f"landmarks={ctrl_rpt['landmarks_found']} | "
              f"subgoals={ctrl_rpt['subgoals_completed']}/{ctrl_rpt['subgoals_generated']} | "
              f"causal={ctrl_rpt['causal_observations']}")

    env.close()
    brain.close()


# ═══════════════════════════════════════════════════════════════════
# MODE 3: GROUND — Analyze human recording, discover RAM semantics
# ═══════════════════════════════════════════════════════════════════

def mode_ground(args):
    """
    Analyze a human recording to discover RAM semantics.

    Reads a .jsonl recording (from 'human' mode), feeds every frame
    through RAMSemanticMapper, and outputs:
    - Which RAM bytes are positions, flags, counters
    - Which bytes correlate with rewards / deaths
    - Entity groups (co-changing position bytes)
    - Subgoal sequences (action chains between reward events)
    """
    print("=" * 60)
    print("MODE: GROUNDING — Analyze Recording")
    print("=" * 60)

    recording_path = args.recording
    if not recording_path:
        # Find most recent recording
        recordings = sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not recordings:
            print("No recordings found. Run 'human' mode first.")
            return
        recording_path = str(recordings[-1])
        print(f"Using most recent recording: {recording_path}")

    print(f"Loading recording from: {recording_path}")

    # Load recording
    frames = []
    with open(recording_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not frames:
        print("No frames found in recording.")
        return

    print(f"  Loaded {len(frames)} frames")

    # Feed through RAM mapper
    mapper = RAMSemanticMapper(ram_size=RAM_SIZE)

    for i, frame in enumerate(frames):
        # HumanRecorder saves as 'ram_hex' (hex-encoded 128 bytes)
        ram_hex = frame.get("ram_hex", "") or frame.get("ram", "")
        action = frame.get("action", 0)
        reward = frame.get("reward", 0.0)
        done = frame.get("done", False)

        # Decode hex RAM
        if isinstance(ram_hex, str) and len(ram_hex) >= 2:
            ram = np.array([int(ram_hex[j:j+2], 16) for j in range(0, len(ram_hex), 2)],
                           dtype=np.uint8)
        elif isinstance(ram_hex, list):
            ram = np.array(ram_hex, dtype=np.uint8)
        else:
            continue

        mapper.observe(ram, action=action, reward=reward, done=done)

    # Get results
    registry = mapper.get_registry()
    subgoals = mapper.get_subgoal_bytes()
    entities = mapper.get_entity_groups()
    report = mapper.report()

    print()
    print("RAM SEMANTIC MAP")
    print("=" * 60)
    print(f"  Active bytes: {report['active_bytes']}")
    print()

    for category, entries in registry.items():
        print(f"  {category.upper()} ({len(entries)} bytes):")
        for entry in entries[:10]:
            addr = entry["addr"]
            print(f"    0x{addr:02X} (byte {addr}): "
                  f"changes={entry.get('change_count', '?')}")
        if len(entries) > 10:
            print(f"    ... and {len(entries) - 10} more")
        print()

    if subgoals:
        print(f"  SUBGOAL BYTES ({len(subgoals)}):")
        for sg in subgoals:
            print(f"    0x{sg['addr']:02X}: changed at reward "
                  f"{sg.get('reward_changes', '?')} times")
        print()

    if entities:
        print(f"  ENTITY GROUPS ({len(entities)}):")
        for i, group in enumerate(entities):
            addrs = [f"0x{a:02X}" for a in group["bytes"]]
            print(f"    Entity {i}: bytes {', '.join(addrs)} "
                  f"(type: {group.get('type', '?')})")
        print()

    # Cross-reference with known addresses
    print("KNOWN ADDRESS VERIFICATION:")
    known = {
        RAM_PLAYER_X: "player_x",
        RAM_PLAYER_Y: "player_y",
        RAM_ROOM: "room",
        RAM_LIVES: "lives",
        RAM_SKULL_X: "skull_x",
        RAM_SKULL_Y: "skull_y",
        RAM_ITEMS: "items",
    }
    for addr, name in known.items():
        discovered = "NOT found"
        for cat, entries in registry.items():
            if any(e["addr"] == addr for e in entries):
                discovered = f"found as {cat}"
                break
        print(f"  0x{addr:02X} ({name:10s}): {discovered}")

    # Save grounding data
    grounding_path = RESULTS_DIR / "grounding.json"
    grounding_data = {
        "source": recording_path,
        "frames_analyzed": len(frames),
        "registry": {k: v for k, v in registry.items()},
        "subgoal_bytes": subgoals,
        "entity_groups": entities,
        "report": report,
    }
    with open(grounding_path, "w") as f:
        json.dump(grounding_data, f, indent=2, default=str)
    print(f"\nGrounding data saved to {grounding_path}")

    # Also extract subgoal sequences for the rehearsal loop
    recorder = HumanRecorder("analysis")
    recorder._frames = frames  # Inject frames
    sequences = recorder.get_subgoal_sequences()
    if sequences:
        print(f"\nSubgoal sequences found: {len(sequences)}")
        for i, seq in enumerate(sequences[:5]):
            print(f"  Seq {i}: {len(seq.get('actions', []))} actions "
                  f"at frame {seq.get('start_frame', '?')}")
    else:
        print("\nNo subgoal sequences found (no reward events in recording)")


# ═══════════════════════════════════════════════════════════════════
# MODE 4: TRAIN — Standard training (no rendering, max speed)
# ═══════════════════════════════════════════════════════════════════

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

    env = make_env(render=False, frameskip=args.frameskip)
    brain = WholeBrain(n_features=RAM_SIZE, n_actions=N_ACTIONS,
                       session_name="montezuma_train", use_cnn=False)
    modules = make_modules()
    meta_planner = MetaPlanner(brain)
    self_model = SelfModel(brain)
    explorer = ExplorationManager(n_actions=N_ACTIONS, decay_episodes=max(args.episodes // 2, 50))
    fall_predictor = FallPredictor()
    controller = IntentionalController(brain, modules["object_graph"], n_actions=N_ACTIONS)
    controller.set_mode("planning")
    recorder = HumanRecorder("train") if args.record else None

    # Load grounding if available
    grounding_path = RESULTS_DIR / "grounding.json"
    if grounding_path.exists():
        print(f"Loading grounding from {grounding_path}")
        with open(grounding_path) as f:
            grounding = json.load(f)
        # Could configure reward_disc from grounding here
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
        prev_ram = ram.copy()
        explorer.on_episode_start()
        fall_predictor.on_episode_start()
        game_state = get_game_state(ram)
        controller.on_episode_start(game_state)

        ep_reward = 0.0
        ep_steps = 0
        ep_max_room = 0
        ep_intrinsic = 0.0
        prev_action = 0
        done = False

        if recorder:
            recorder.start(None)

        while not done and ep_steps < args.max_steps:
            ram = env.unwrapped.ale.getRAM()
            game_state = get_game_state(ram)
            ep_max_room = max(ep_max_room, game_state["room"])

            modules["ram_mapper"].observe(ram, action=prev_action, reward=ep_reward, done=done)
            modules["temporal"].observe(ram, step=total_steps)

            intrinsic_r = modules["reward_disc"].compute(
                prev_ram, ram, action=prev_action, done=done,
            )
            # Add position novelty bonus
            intrinsic_r += explorer.get_novelty_bonus(game_state)
            ep_intrinsic += intrinsic_r

            modules["object_graph"].add_entity("player", {
                "x": game_state["player_x"], "y": game_state["player_y"],
            }, category="player")
            modules["object_graph"].add_entity("skull", {
                "x": game_state["skull_x"], "y": game_state["skull_y"],
            }, category="enemy")

            features = ram.astype(np.float32) / 255.0
            result = brain.step(features, prev_action=prev_action, reward=intrinsic_r, done=False)
            brain_action = result["action"] if isinstance(result, dict) else result

            # ── Intentional Controller: subgoals + skills + shaping ──
            ctrl_result = controller.step(
                game_state, features, brain_action,
                reward=ep_reward, done=done,
            )
            action = ctrl_result["action"]
            intrinsic_r += ctrl_result["shaped_reward"]

            # Predictive fall avoidance: filter actions BEFORE selection
            fall_safe = fall_predictor.filter_actions(game_state, list(range(N_ACTIONS)))

            # Exploration override (safety net)
            action = explorer.select_action(action, game_state, features)

            safe_actions = modules["safety"].filter_actions(
                features, fall_safe,
                player_pos=(game_state["player_x"], game_state["player_y"]),
            )
            if action not in safe_actions:
                action = safe_actions[0] if safe_actions else action

            meta_planner.observe(features, reward=intrinsic_r, done=False, action=action)
            modules["proc_memory"].observe_transition(action, features, features, intrinsic_r)

            if recorder:
                recorder.record(ram, action, ep_reward, False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            prev_action = action
            prev_ram = ram.copy()

            # Learn from falls (after step, so we see the y-delta)
            fall_predictor.observe(get_game_state(env.unwrapped.ale.getRAM()), action)

            self_model.record_position(
                game_state["player_x"], game_state["player_y"],
                area_hash=game_state["room"],
            )
            if done and reward <= 0:
                modules["safety"].learn_from_death(features, prev_action, context="death")
                self_model.record_death("death", features)

        brain.step(ram.astype(np.float32) / 255.0,
                   prev_action=prev_action, reward=ep_reward, done=True)
        meta_planner.observe(ram.astype(np.float32) / 255.0, reward=ep_reward, done=True)

        if recorder:
            recorder.stop()

        episode_rewards.append(ep_reward)
        best_reward = max(best_reward, ep_reward)
        best_room = max(best_room, ep_max_room)

        mode = meta_planner.decide()
        # Sync MetaPlanner mode → controller mode
        if mode in ("planning", "explore"):
            controller.set_mode(mode)
        elif mode == "reactive":
            controller.set_mode("reactive")

        elapsed = time.time() - start_time
        fps = total_steps / max(elapsed, 1)

        if episode % 10 == 0 or ep_reward > 0 or args.verbose:
            avg_r = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            ctrl_rpt = controller.report()
            print(f"Ep {episode:4d} | R={ep_reward:6.0f} | avg50={avg_r:6.1f} | "
                  f"best={best_reward:6.0f} | room={ep_max_room}(best={best_room}) | "
                  f"steps={ep_steps:5d} | intr={ep_intrinsic:6.2f} | "
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
        rec_path = recorder.save(str(RECORDINGS_DIR))
        print(f"  Training recording saved to {rec_path}")

    env.close()
    brain.close()


# ═══════════════════════════════════════════════════════════════════
# MODE 5: PLAN — Goal-directed training with subgoal planner
# ═══════════════════════════════════════════════════════════════════

def mode_plan(args):
    """
    Goal-directed training with the SubgoalPlanner active.

    Similar to 'train' but the MetaPlanner is forced into planning
    mode. The agent uses the landmark graph + goal regression +
    causal model to navigate with intentional subgoals rather than
    random exploration.

    Best used AFTER 'ground' mode has discovered RAM semantics.
    """
    print("=" * 60)
    print("MODE: PLAN (goal-directed, SubgoalPlanner active)")
    print("=" * 60)
    print()
    print("MetaPlanner forced to 'planning' mode.")
    print("Agent will use landmark graph + goal regression.")
    print()

    # Same as train but with forced planning mode
    env = make_env(render=args.render, frameskip=args.frameskip)
    brain = WholeBrain(n_features=RAM_SIZE, n_actions=N_ACTIONS,
                       session_name="montezuma_plan", use_cnn=False)
    modules = make_modules()
    meta_planner = MetaPlanner(brain)
    self_model = SelfModel(brain)

    # Force planning mode
    meta_planner.force_mode("planning")

    episode_rewards = []
    best_reward = 0.0
    best_room = 0
    total_steps = 0
    start_time = time.time()

    for episode in range(args.episodes):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()
        prev_ram = ram.copy()

        ep_reward = 0.0
        ep_steps = 0
        ep_max_room = 0
        prev_action = 0
        done = False

        while not done and ep_steps < args.max_steps:
            ram = env.unwrapped.ale.getRAM()
            game_state = get_game_state(ram)
            ep_max_room = max(ep_max_room, game_state["room"])

            modules["ram_mapper"].observe(ram, action=prev_action, reward=ep_reward, done=done)

            intrinsic_r = modules["reward_disc"].compute(
                prev_ram, ram, action=prev_action, done=done,
            )

            features = ram.astype(np.float32) / 255.0
            result = brain.step(features, prev_action=prev_action, reward=intrinsic_r, done=False)
            action = result["action"] if isinstance(result, dict) else result

            safe_actions = modules["safety"].filter_actions(
                features, list(range(N_ACTIONS)),
                player_pos=(game_state["player_x"], game_state["player_y"]),
            )
            if action not in safe_actions:
                action = safe_actions[0] if safe_actions else action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            prev_action = action
            prev_ram = ram.copy()

            if done and reward <= 0:
                modules["safety"].learn_from_death(features, prev_action)
                self_model.record_death("death")

        brain.step(ram.astype(np.float32) / 255.0,
                   prev_action=prev_action, reward=ep_reward, done=True)

        episode_rewards.append(ep_reward)
        best_reward = max(best_reward, ep_reward)
        best_room = max(best_room, ep_max_room)

        if episode % 10 == 0 or ep_reward > 0 or args.verbose:
            avg_r = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1)
            print(f"Ep {episode:4d} | R={ep_reward:6.0f} | avg50={avg_r:6.1f} | "
                  f"room={ep_max_room}(best={best_room}) | "
                  f"steps={ep_steps:5d} | {fps:5.0f} fps | mode=PLANNING")

    elapsed = time.time() - start_time
    _print_final_report(args.episodes, total_steps, best_reward, best_room,
                        episode_rewards, elapsed, modules, meta_planner, self_model)
    env.close()
    brain.close()


# ═══════════════════════════════════════════════════════════════════
# MODE 6: REHEARSE — Practice bottleneck areas with save states
# ═══════════════════════════════════════════════════════════════════

def mode_rehearse(args):
    """
    Focused practice on bottleneck areas.

    Uses save/load state (ALE snapshots) to repeatedly practice
    difficult sections. The rehearsal loop identifies where the
    agent dies most and creates targeted practice sessions.

    Modes:
      advance  — Pause-verify-execute chain building
      frontier — Play from start, advance on death
      stuck    — 10 failures → train flanking areas
      free     — Full exploration with save states
    """
    print("=" * 60)
    print(f"MODE: REHEARSE ({args.rehearse_mode})")
    print("=" * 60)

    env = make_env(render=args.render, frameskip=args.frameskip)
    brain = WholeBrain(n_features=RAM_SIZE, n_actions=N_ACTIONS,
                       session_name="montezuma_rehearse", use_cnn=False)
    modules = make_modules()

    # Force meta-planner into rehearse mode
    meta_planner = MetaPlanner(brain)
    meta_planner.force_mode("rehearse")

    # First: run a few episodes to identify bottlenecks
    print("\nPhase 1: Identify bottlenecks (10 episodes)...")
    for ep in range(10):
        obs, info = env.reset()
        ram = env.unwrapped.ale.getRAM()
        prev_ram = ram.copy()
        ep_steps = 0
        prev_action = 0
        done = False

        while not done and ep_steps < args.max_steps:
            ram = env.unwrapped.ale.getRAM()
            features = ram.astype(np.float32) / 255.0
            intrinsic_r = modules["reward_disc"].compute(prev_ram, ram, action=prev_action, done=done)

            result = brain.step(features, prev_action=prev_action, reward=intrinsic_r, done=False)
            action = result["action"] if isinstance(result, dict) else result

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1
            prev_action = action
            prev_ram = ram.copy()

            if done and reward <= 0:
                modules["safety"].learn_from_death(features, prev_action, context="death")

        brain.step(ram.astype(np.float32) / 255.0,
                   prev_action=prev_action, reward=0.0, done=True)

    # Phase 2: Rehearsal
    print(f"\nPhase 2: Rehearse ({args.rehearse_mode} mode, {args.episodes} rounds)...")

    try:
        for round_num in range(args.episodes):
            obs, info = env.reset()
            features = env.unwrapped.ale.getRAM().astype(np.float32) / 255.0
            result = brain.rehearse(mode=args.rehearse_mode, env=env, features=features)
            print(f"  Round {round_num}: {result}")
    except Exception as e:
        print(f"  Rehearsal error: {e}")
        print("  (Rehearsal requires save/load state support in env)")

    env.close()
    brain.close()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _save_stats(episode, total_steps, best_reward, best_room,
                episode_rewards, elapsed, mode, modules, meta_planner,
                self_model, brain):
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
    stats_path = RESULTS_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def _print_final_report(n_episodes, total_steps, best_reward, best_room,
                         episode_rewards, elapsed, modules, meta_planner, self_model):
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


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Montezuma's Revenge — Full WholeBrain Stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  human     Play yourself with keyboard, record RAM for grounding
  watch     Watch the agent play (rendered)
  ground    Analyze a human recording to discover RAM semantics
  train     Standard training (no rendering, max speed)
  plan      Goal-directed training with SubgoalPlanner
  rehearse  Focused bottleneck practice with save states

RECOMMENDED WORKFLOW:
  1. python run_montezuma_brain.py human --episodes 3
     (Play 3 episodes yourself, record RAM)

  2. python run_montezuma_brain.py ground
     (Analyze your recording, discover game objects)

  3. python run_montezuma_brain.py train --episodes 500
     (Train with grounding data, all modules active)

  4. python run_montezuma_brain.py watch --episodes 5
     (Watch how the agent plays after training)

  5. python run_montezuma_brain.py rehearse --rehearse-mode stuck
     (Practice the specific areas where agent keeps dying)
""",
    )

    parser.add_argument("mode", choices=["human", "watch", "ground", "train", "plan", "rehearse"],
                        help="Operating mode")
    parser.add_argument("recording", nargs="?", default=None,
                        help="Path to recording file (for 'ground' mode)")

    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes (default: 500)")
    parser.add_argument("--max-steps", type=int, default=27000, help="Max steps per episode")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--record", action="store_true", help="Record RAM during training")
    parser.add_argument("--render", action="store_true", help="Render during plan/rehearse modes")
    parser.add_argument("--save-freq", type=int, default=25, help="Save stats every N episodes")
    parser.add_argument("--rehearse-mode", choices=["advance", "frontier", "stuck", "free"],
                        default="advance", help="Rehearsal sub-mode (default: advance)")

    args = parser.parse_args()

    if args.mode == "human":
        mode_human(args)
    elif args.mode == "watch":
        mode_watch(args)
    elif args.mode == "ground":
        mode_ground(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "plan":
        mode_plan(args)
    elif args.mode == "rehearse":
        mode_rehearse(args)
