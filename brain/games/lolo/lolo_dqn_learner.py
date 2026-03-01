"""
lolo_dqn_learner.py — DQN learner for Lolo using compressed 84-dim state.

Replaces tabular SARSA with a small neural network that generalizes
across states. The network maps compressed_state(84) → Q(6 actions).

Architecture:
  - Input: 84-dim compressed state (from LoloCompressedState)
  - Hidden: 128 → 64 (ReLU)
  - Output: 6 action Q-values (UP, DOWN, LEFT, RIGHT, SHOOT, WAIT)

Training:
  - Experience replay (prioritized by |TD-error|)
  - Target network (soft update every N steps)
  - Epsilon-greedy exploration (decaying)

Transfer:
  - Save/load as .pt file
  - Same compressed state works for sim and NES RAM
  - Wires into basal ganglia as habit network
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from brain.games.lolo.lolo_compressed_state import LoloCompressedState


# ── Q-Network ────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Small MLP: state(84) → Q(6)."""

    def __init__(self, state_dim: int = 84, n_actions: int = 6,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────

class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self._capacity = capacity
        self._alpha = alpha
        self._buffer: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)
        self.total_stored = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, td_error: float = 1.0):
        priority = (abs(td_error) + 0.01) ** self._alpha
        if abs(reward) > 1.0:
            priority *= 3.0
        if done and reward > 5.0:
            priority *= 5.0
        self._buffer.append((state, action, reward, next_state, done))
        self._priorities.append(priority)
        self.total_stored += 1

    def sample(self, batch_size: int) -> Tuple:
        n = len(self._buffer)
        if n < batch_size:
            batch_size = n

        priorities = np.array(list(self._priorities), dtype=np.float32)
        probs = priorities / (priorities.sum() + 1e-10)
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self._buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self._buffer)


# ── DQN Learner ──────────────────────────────────────────────────────

class LoloDQNLearner:
    """
    DQN learner for Lolo with compressed 84-dim state.

    Drop-in replacement for LoloSarsaLearner — same API:
      - set_simulator(sim)
      - step(obs, prev_action, reward, done) → {"action": int}
      - report() → dict
      - get_transfer_data() → dict
    """

    def __init__(
        self,
        n_actions: int = 6,
        state_dim: int = 84,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        replay_capacity: int = 100000,
        target_update_freq: int = 200,
        tau: float = 0.01,
        learn_every: int = 4,
        min_replay: int = 256,
    ):
        if not HAS_TORCH:
            raise ImportError("DQN requires PyTorch: pip install torch")

        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.learn_every = learn_every
        self.min_replay = min_replay

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.memory = ReplayBuffer(capacity=replay_capacity)

        # Compressed state encoder
        self._encoder = LoloCompressedState()

        # Simulator reference
        self._sim = None

        # Tracking
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: int = 0
        self._total_steps = 0
        self._total_episodes = 0
        self._total_learns = 0
        self._episode_reward = 0.0
        self._recent_losses: deque = deque(maxlen=100)

        # Success chains for transfer
        self.success_chains: List[List[int]] = []
        self._current_chain: List[int] = []

    def set_simulator(self, sim):
        """Set simulator for direct state access."""
        self._sim = sim

    def set_puzzle_id(self, pid: int):
        """API compat — not used by DQN."""
        pass

    def _get_state(self) -> np.ndarray:
        """Get compressed 84-dim state from simulator."""
        if self._sim is not None:
            return self._encoder.encode_from_sim(self._sim)
        return np.zeros(self.state_dim, dtype=np.float32)

    def _select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            q_values = self.q_net(s)
            return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (for basal ganglia integration)."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return self.q_net(s).cpu().numpy().flatten()

    def step(
        self,
        obs: Any = None,
        prev_action: int = 0,
        reward: float = 0.0,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Same API as LoloSarsaLearner.step()."""
        state = self._get_state()
        self._total_steps += 1
        self._episode_reward += reward

        action = self._select_action(state)

        # Store transition
        if self._prev_state is not None:
            self.memory.push(
                state=self._prev_state,
                action=self._prev_action,
                reward=reward,
                next_state=state,
                done=done,
            )

        self._current_chain.append(action)

        # Learn from replay
        if (self._total_steps % self.learn_every == 0
                and len(self.memory) >= self.min_replay):
            self._learn()

        # Soft update target network
        if self._total_steps % self.target_update_freq == 0:
            self._soft_update()

        # Episode boundary
        if done:
            self._total_episodes += 1
            self.epsilon = max(self.epsilon_end,
                               self.epsilon * self.epsilon_decay)

            if self._episode_reward > 5:
                if len(self._current_chain) < 500:
                    self.success_chains.append(self._current_chain.copy())
                    if len(self.success_chains) > 200:
                        self.success_chains = self.success_chains[-200:]

            self._current_chain = []
            self._episode_reward = 0.0
            self._prev_state = None
            self._prev_action = 0
        else:
            self._prev_state = state.copy()
            self._prev_action = action

        return {"action": action}

    def _learn(self):
        """One gradient step from replay buffer."""
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        s = torch.tensor(states, device=self.device)
        a = torch.tensor(actions, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, device=self.device)
        ns = torch.tensor(next_states, device=self.device)
        d = torch.tensor(dones, device=self.device)

        # Current Q
        q_current = self.q_net(s).gather(1, a).squeeze(1)

        # Target Q (Double DQN: select action with online, evaluate with target)
        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
            q_target = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target = r + self.gamma * q_target * (1.0 - d)

        loss = self.loss_fn(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._total_learns += 1
        self._recent_losses.append(loss.item())

    def _soft_update(self):
        """Soft update target network: θ_target = τ·θ_online + (1-τ)·θ_target."""
        for tp, op in zip(self.target_net.parameters(),
                          self.q_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save DQN weights and training state."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "total_learns": self._total_learns,
        }, path)

    def load(self, path: str):
        """Load DQN weights and resume training state."""
        checkpoint = torch.load(path, weights_only=False,
                                map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self._total_steps = checkpoint.get("total_steps", 0)
        self._total_episodes = checkpoint.get("total_episodes", 0)
        self._total_learns = checkpoint.get("total_learns", 0)

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        avg_loss = (float(np.mean(self._recent_losses))
                    if self._recent_losses else 0.0)
        return {
            "type": "LoloDQNLearner",
            "device": str(self.device),
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "total_learns": self._total_learns,
            "epsilon": round(self.epsilon, 4),
            "avg_loss": round(avg_loss, 6),
            "replay_size": len(self.memory),
            "success_chains": len(self.success_chains),
            "q_net_params": sum(p.numel() for p in self.q_net.parameters()),
        }

    def close(self):
        pass

    def get_transfer_data(self) -> Dict[str, Any]:
        """Extract learned knowledge for transfer to full brain."""
        return {
            "chains": self.success_chains[-50:],
            "stats": self.report(),
        }

    # ── Q-table compatibility (for GAN trainer) ──────────────────────

    @property
    def q_table(self) -> dict:
        """Fake Q-table property for GAN trainer compat. Returns size proxy."""
        return {"_dqn_size": self._total_learns}

    @q_table.setter
    def q_table(self, value):
        """Ignore Q-table assignment (compat)."""
        pass
