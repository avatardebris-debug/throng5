"""
lolo_fast_learner.py — Lightweight Q-learner for rapid Lolo training.

Uses simulator-direct state: (puzzle_id, player_row, player_col, hearts_collected)
Only ~700 possible states per puzzle. Proven 96.2% on single puzzle.

Usage with curriculum (fast path):
    learner = LoloFastLearner(n_actions=6)
    learner.set_simulator(sim)     # Direct sim access for state
    learner.set_puzzle_id(0)
    result = learner.step(obs, reward=r, done=d)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np


class LoloFastLearner:
    """
    Tabular Q-learner using direct simulator state.
    
    State = (puzzle_id, player_row, player_col, hearts_collected)
    ~700 states per puzzle, 40K+ steps/sec.
    """

    def __init__(
        self,
        n_actions: int = 6,
        lr: float = 0.3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.998,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[tuple, np.ndarray] = {}
        self._sim = None          # Direct simulator reference
        self._puzzle_id = 0
        self.success_chains: List[List[int]] = []
        self._current_chain: List[int] = []

        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._prev_state: Optional[tuple] = None

    def set_simulator(self, sim):
        """Set simulator reference for direct state access."""
        self._sim = sim

    def set_puzzle_id(self, pid: int):
        self._puzzle_id = pid

    def _get_state(self) -> tuple:
        """Read state directly from simulator — no float parsing."""
        if self._sim is not None:
            return (self._puzzle_id,
                    self._sim.player_row,
                    self._sim.player_col,
                    self._sim.hearts_collected)
        return (self._puzzle_id, 0, 0, 0)

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state]

    def step(
        self,
        obs: Any = None,
        prev_action: int = 0,
        reward: float = 0.0,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Same API as WholeBrain.step(). obs is ignored — uses sim directly."""
        state = self._get_state()
        self._total_steps += 1
        self._episode_reward += reward

        # Q-learning update
        if self._prev_state is not None:
            q_prev = self._get_q(self._prev_state)
            target = reward if done else reward + self.gamma * np.max(self._get_q(state))
            q_prev[prev_action] += self.lr * (target - q_prev[prev_action])

        # ε-greedy
        q_values = self._get_q(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(q_values))

        self._current_chain.append(action)

        if done:
            self._total_episodes += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if self._episode_reward > 5:
                if len(self._current_chain) < 500:
                    self.success_chains.append(self._current_chain.copy())
                    if len(self.success_chains) > 100:
                        self.success_chains = self.success_chains[-100:]
            self._current_chain = []
            self._episode_reward = 0.0
            self._prev_state = None
        else:
            self._prev_state = state

        return {"action": action}

    def report(self) -> Dict[str, Any]:
        return {
            "type": "LoloFastLearner",
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "q_table_size": len(self.q_table),
            "success_chains": len(self.success_chains),
            "epsilon": round(self.epsilon, 4),
        }

    def close(self):
        pass

    def get_transfer_data(self) -> Dict[str, Any]:
        top = []
        for s, q in self.q_table.items():
            v = float(np.max(q))
            if v > 0:
                top.append({"state": s, "action": int(np.argmax(q)), "value": v})
        top.sort(key=lambda x: x["value"], reverse=True)
        return {"q_summary": top[:500], "chains": self.success_chains[-50:], "stats": self.report()}
