"""Montezuma exploration epsilon and anti-stuck filtering."""

from __future__ import annotations

import numpy as np


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
