"""
lolo_sarsa_learner.py — SARSA with integrated hippocampus-style replay.

Primary fast learner for Lolo curriculum training.
Uses on-policy SARSA with a prioritized replay buffer that periodically
re-trains the Q-table on old transitions to prevent catastrophic forgetting
when scaling to many puzzles.

Performance: ~40K sps on easy puzzles, 88% success on 5 fixed puzzles.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ReplayMemory:
    """
    Lightweight prioritized replay buffer (hippocampus-style).

    Stores (state_key, action, reward, next_state_key, done) transitions.
    Samples prioritized by |reward| and recency.
    No MessageBus or BrainRegion dependency — pure data structure.
    """

    def __init__(self, capacity: int = 50000, priority_alpha: float = 0.6):
        self._capacity = capacity
        self._alpha = priority_alpha
        self._transitions: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)
        self._total_stored = 0
        self._total_replayed = 0

    def store(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
        td_error: float = 0.0,
    ) -> None:
        """Store one transition with priority."""
        priority = (abs(td_error) + 0.01) ** self._alpha
        if abs(reward) > 1.0:
            priority *= 3.0  # Boost high-reward transitions
        if done and reward > 5.0:
            priority *= 5.0  # Strongly prioritize wins

        self._transitions.append((state, action, reward, next_state, done))
        self._priorities.append(priority)
        self._total_stored += 1

    def sample(self, batch_size: int = 32) -> List[Tuple]:
        """Sample a prioritized batch."""
        n = len(self._transitions)
        if n < batch_size:
            return list(self._transitions)

        priorities = np.array(list(self._priorities), dtype=np.float32)
        probs = priorities / (priorities.sum() + 1e-10)
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        self._total_replayed += batch_size
        return [self._transitions[i] for i in indices]

    def __len__(self):
        return len(self._transitions)

    def stats(self) -> Dict[str, int]:
        return {
            "stored": self._total_stored,
            "replayed": self._total_replayed,
            "buffer_size": len(self._transitions),
        }


from brain.games.lolo.lolo_compressed_state import LoloCompressedState


class LoloSarsaLearner:
    """
    On-policy SARSA with hippocampus-style replay for catastrophic forgetting prevention.

    Architecture:
      - Tabular Q-values keyed by compressed 84-dim state (quantized to 1 decimal)
      - On-policy SARSA updates for current transitions
      - Periodic off-policy replay from memory buffer (every N episodes)
      - Prioritized by |reward| and success (wins get 5x replay priority)

    State = compressed 84-dim tuple (local grid, enemies, danger, path info)
    Generalizes across puzzles — same compressed state = same Q-values.
    """

    def __init__(
        self,
        n_actions: int = 6,
        lr: float = 0.3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.998,
        replay_capacity: int = 50000,
        replay_batch: int = 64,
        replay_interval: int = 5,  # Replay every N episodes
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_interval = replay_interval
        self.replay_batch = replay_batch

        # Q-table: compressed_state_tuple → action → value
        self.q_table: Dict[tuple, np.ndarray] = {}

        # Compressed state encoder (same for sim and NES RAM)
        self._encoder = LoloCompressedState()

        # Hippocampus-style replay memory
        self.memory = ReplayMemory(
            capacity=replay_capacity,
            priority_alpha=0.6,
        )

        # Simulator reference (set by curriculum)
        self._sim = None

        # Episode tracking
        self._prev_state: Optional[tuple] = None
        self._prev_action: int = 0
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0

        # Transfer artifacts
        self.success_chains: List[List[int]] = []
        self._current_chain: List[int] = []

    def set_simulator(self, sim):
        """Set simulator reference for direct state access."""
        self._sim = sim

    def set_puzzle_id(self, pid: int):
        """Kept for API compat — puzzle_id no longer used in state key."""
        pass

    def _get_state(self) -> tuple:
        """Compressed 84-dim state — generalizes across puzzles."""
        if self._sim is not None:
            return self._encoder.encode_key(self._sim)
        return tuple(np.zeros(84, dtype=np.float32))

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state]

    def _select_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(state)))

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

        # Select action FIRST (on-policy for SARSA)
        action = self._select_action(state)

        # SARSA update: Q(s,a) += lr * (r + gamma * Q(s', a') - Q(s,a))
        if self._prev_state is not None:
            q_prev = self._get_q(self._prev_state)
            if done:
                target = reward
            else:
                # SARSA: use the ACTUAL next action's Q (not max)
                target = reward + self.gamma * self._get_q(state)[action]

            td_error = target - q_prev[self._prev_action]
            q_prev[self._prev_action] += self.lr * td_error

            # Store in replay memory
            self.memory.store(
                state=self._prev_state,
                action=self._prev_action,
                reward=reward,
                next_state=state,
                done=done,
                td_error=td_error,
            )

        self._current_chain.append(action)

        # Episode boundary
        if done:
            self._total_episodes += 1
            self.epsilon = max(self.epsilon_end,
                              self.epsilon * self.epsilon_decay)

            # Store success chains
            if self._episode_reward > 5:
                if len(self._current_chain) < 500:
                    self.success_chains.append(self._current_chain.copy())
                    if len(self.success_chains) > 200:
                        self.success_chains = self.success_chains[-200:]

            # Periodic replay from memory (prevent forgetting)
            if (self._total_episodes % self.replay_interval == 0
                    and len(self.memory) >= self.replay_batch):
                self._replay()

            self._current_chain = []
            self._episode_reward = 0.0
            self._prev_state = None
            self._prev_action = 0
        else:
            self._prev_state = state
            self._prev_action = action

        return {"action": action}

    def _replay(self) -> None:
        """
        Replay transitions from memory buffer (off-policy Q-learning style).

        Uses Q-learning (max Q) for replay since we don't have the original
        on-policy action available. This is standard practice for combined
        SARSA + experience replay.
        """
        batch = self.memory.sample(self.replay_batch)
        replay_lr = self.lr * 0.5  # Lower LR for replay stability

        for state, action, reward, next_state, done in batch:
            q = self._get_q(state)
            if done:
                target = reward
            else:
                # Q-learning for replay (max Q): more stable than SARSA for off-policy
                target = reward + self.gamma * np.max(self._get_q(next_state))
            q[action] += replay_lr * (target - q[action])

    def report(self) -> Dict[str, Any]:
        return {
            "type": "LoloSarsaLearner",
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "q_table_size": len(self.q_table),
            "success_chains": len(self.success_chains),
            "epsilon": round(self.epsilon, 4),
            "memory": self.memory.stats(),
        }

    def close(self):
        pass

    def get_transfer_data(self) -> Dict[str, Any]:
        """Extract learned knowledge for transfer to full brain."""
        top = []
        for s, q in self.q_table.items():
            v = float(np.max(q))
            if v > 0:
                top.append({"state": s, "action": int(np.argmax(q)), "value": v})
        top.sort(key=lambda x: x["value"], reverse=True)
        return {
            "q_summary": top[:500],
            "chains": self.success_chains[-50:],
            "stats": self.report(),
        }
