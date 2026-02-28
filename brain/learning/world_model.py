"""
world_model.py — Learned World Model for Dream Simulation.

Predicts:
  - Next state features given current features + action
  - Expected reward for that transition

Architecture:
  Input:  features (84) + action_onehot (n_actions) = 84 + n_actions
  Hidden: 2 layers of 256 with LayerNorm + ReLU
  Output heads:
    - State predictor: 84-dim (residual: predicts delta_features)
    - Reward predictor: 1-dim (scalar reward prediction)

The world model learns from real transitions stored in replay.
The dreamer uses it to simulate multi-step futures through imagination.

Usage:
    wm = WorldModel(n_features=84, n_actions=6)
    wm.train_batch(states, actions, next_states, rewards)
    dreams = wm.dream(start_features, horizon=5, policy_fn=dqn.select_action)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WorldModel:
    """
    Learned dynamics model for dream simulation.

    Predicts (next_state, reward) given (state, action).
    Uses residual state prediction: next_state = state + delta.
    Trains from real transitions, then the dreamer "imagines"
    future trajectories by chaining predictions.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_size: int = 256,
        lr: float = 1e-3,
        buffer_size: int = 50000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.batch_size = batch_size
        self._total_updates = 0
        self._losses: deque = deque(maxlen=100)
        self._replay: deque = deque(maxlen=buffer_size)

        if not TORCH_AVAILABLE:
            self._net = None
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = n_features + n_actions

        # ── Shared encoder ────────────────────────────────────────────
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # ── State prediction head (predicts DELTA, not absolute) ──────
        self._state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_features),
        ).to(self.device)

        # ── Reward prediction head ────────────────────────────────────
        self._reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        ).to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────
        all_params = (
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters())
        )
        self._optimizer = optim.Adam(all_params, lr=lr)

    @property
    def is_ready(self) -> bool:
        """True if model has trained enough to produce useful predictions."""
        return self._total_updates >= 10

    @property
    def confidence(self) -> float:
        """0-1 confidence based on training progress and loss."""
        if self._total_updates == 0:
            return 0.0
        train_conf = min(1.0, self._total_updates / 100.0)
        if self._losses:
            loss_conf = max(0.0, 1.0 - float(np.mean(self._losses)))
        else:
            loss_conf = 0.0
        return train_conf * 0.5 + loss_conf * 0.5

    # ── Training ──────────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
    ) -> None:
        """Store a real transition for training."""
        self._replay.append((
            np.asarray(state, dtype=np.float32),
            action,
            np.asarray(next_state, dtype=np.float32),
            reward,
        ))

    def train_step(self) -> Dict[str, float]:
        """Train on a batch of real transitions."""
        if not TORCH_AVAILABLE or self._encoder is None:
            return {"wm_loss": 0.0}

        if len(self._replay) < self.batch_size:
            return {"wm_loss": 0.0, "wm_buffer": len(self._replay)}

        # Sample batch
        indices = np.random.choice(len(self._replay), self.batch_size, replace=False)
        batch = [self._replay[i] for i in indices]

        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions_idx = [b[1] for b in batch]
        next_states = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        rewards = torch.FloatTensor([b[3] for b in batch]).unsqueeze(1).to(self.device)

        # One-hot encode actions
        actions_oh = torch.zeros(self.batch_size, self.n_actions).to(self.device)
        for i, a in enumerate(actions_idx):
            actions_oh[i, a] = 1.0

        # Forward
        x = torch.cat([states, actions_oh], dim=1)
        encoded = self._encoder(x)

        # Predict delta (residual) and reward
        delta_pred = self._state_head(encoded)
        reward_pred = self._reward_head(encoded)

        # Targets
        delta_target = next_states - states  # Residual target
        reward_target = rewards

        # Losses
        state_loss = F.mse_loss(delta_pred, delta_target)
        reward_loss = F.smooth_l1_loss(reward_pred, reward_target)
        total_loss = state_loss + reward_loss * 0.1  # Reward is secondary signal

        # Optimize
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters()),
            5.0,
        )
        self._optimizer.step()

        self._total_updates += 1
        loss_val = total_loss.item()
        self._losses.append(loss_val)

        return {
            "wm_loss": loss_val,
            "wm_state_loss": state_loss.item(),
            "wm_reward_loss": reward_loss.item(),
            "wm_updates": self._total_updates,
            "wm_buffer": len(self._replay),
        }

    # ── Prediction ────────────────────────────────────────────────────

    def predict(
        self, features: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward for a single (state, action) pair.

        Returns: (predicted_next_features, predicted_reward)
        """
        if not TORCH_AVAILABLE or self._encoder is None:
            return features.copy(), 0.0

        with torch.inference_mode():
            state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_oh = torch.zeros(1, self.n_actions, device=self.device)
            action_oh[0, action] = 1.0

            x = torch.cat([state_t, action_oh], dim=1)
            encoded = self._encoder(x)
            delta = self._state_head(encoded)
            reward = self._reward_head(encoded)

            next_features = (state_t + delta).squeeze(0).cpu().numpy()
            pred_reward = reward.item()

        return next_features, pred_reward

    # ── Dreaming ──────────────────────────────────────────────────────

    def dream(
        self,
        start_features: np.ndarray,
        horizon: int = 5,
        policy_fn: Optional[Callable] = None,
        n_actions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulate a trajectory through imagination.

        Args:
            start_features: Current real state features (84-dim)
            horizon: How many steps to simulate forward
            policy_fn: Callable(features) -> (action, q_values)
                       If None, tries all actions and picks best
            n_actions: Override for action count

        Returns:
            List of dream steps, each containing:
                features: imagined state features
                action: action taken in dream
                predicted_reward: model's reward prediction
                predicted_value: DQN's Q-value of imagined state (if policy_fn)
                step: dream step index
        """
        if not self.is_ready:
            return []

        n_act = n_actions or self.n_actions
        features = start_features.copy()
        trajectory = []

        for step in range(horizon):
            if policy_fn is not None:
                # Use DQN policy to pick dream action
                action, q_values = policy_fn(features, explore=False)
                value = float(q_values[action])
            else:
                # Try all actions, pick the one with best predicted reward
                best_action = 0
                best_reward = -float("inf")
                for a in range(n_act):
                    _, pred_r = self.predict(features, a)
                    if pred_r > best_reward:
                        best_reward = pred_r
                        best_action = a
                action = best_action
                value = best_reward

            # Simulate step
            next_features, pred_reward = self.predict(features, action)

            trajectory.append({
                "features": features.copy(),
                "action": action,
                "predicted_reward": pred_reward,
                "predicted_value": value,
                "step": step,
            })

            features = next_features

        return trajectory

    def dream_all_actions(
        self,
        features: np.ndarray,
        depth: int = 3,
        gamma: float = 0.99,
    ) -> np.ndarray:
        """
        Evaluate all actions by dreaming depth steps ahead.

        Returns: action_values (n_actions,) — estimated return for each action.
        """
        if not self.is_ready:
            return np.zeros(self.n_actions)

        action_values = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Simulate first step
            next_feat, r = self.predict(features, a)
            total_return = r

            # Continue greedily for remaining steps
            feat = next_feat
            for d in range(1, depth):
                # Pick best action from remaining
                best_r = -float("inf")
                best_a = 0
                for a2 in range(self.n_actions):
                    _, pr = self.predict(feat, a2)
                    if pr > best_r:
                        best_r = pr
                        best_a = a2
                feat, r = self.predict(feat, best_a)
                total_return += (gamma ** d) * r

            action_values[a] = total_return

        return action_values

    # ── Stats ─────────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        n_params = 0
        if TORCH_AVAILABLE and self._encoder is not None:
            n_params = (
                sum(p.numel() for p in self._encoder.parameters())
                + sum(p.numel() for p in self._state_head.parameters())
                + sum(p.numel() for p in self._reward_head.parameters())
            )
        return {
            "wm_params": n_params,
            "wm_updates": self._total_updates,
            "wm_buffer": len(self._replay),
            "wm_confidence": round(self.confidence, 3),
            "wm_is_ready": self.is_ready,
            "wm_avg_loss": round(float(np.mean(self._losses)), 6) if self._losses else 0.0,
        }

    # ── Checkpoint persistence ────────────────────────────────────────

    def save_state(self) -> Dict[str, Any]:
        """Serialize world model state for checkpointing."""
        state = {
            "total_updates": self._total_updates,
            "losses": list(self._losses),
            "n_features": self.n_features,
            "n_actions": self.n_actions,
        }
        if TORCH_AVAILABLE and self._encoder is not None:
            state["encoder"] = self._encoder.state_dict()
            state["state_head"] = self._state_head.state_dict()
            state["reward_head"] = self._reward_head.state_dict()
            state["optimizer"] = self._optimizer.state_dict()
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore world model from checkpoint state."""
        self._total_updates = state.get("total_updates", 0)
        losses = state.get("losses", [])
        self._losses = deque(losses, maxlen=100)

        if TORCH_AVAILABLE and self._encoder is not None:
            if "encoder" in state:
                self._encoder.load_state_dict(state["encoder"])
            if "state_head" in state:
                self._state_head.load_state_dict(state["state_head"])
            if "reward_head" in state:
                self._reward_head.load_state_dict(state["reward_head"])
            if "optimizer" in state:
                self._optimizer.load_state_dict(state["optimizer"])

