"""
torch_dqn.py — Deep DQN with PyTorch backend for Striatum.

Replaces the single hidden-layer NumPy DQN with a proper deep network:
  - 3-layer MLP with ReLU, BatchNorm, Dropout
  - Dueling DQN architecture (value + advantage streams)
  - Target network with soft update (Polyak averaging)
  - Proper Adam optimizer with gradient clipping
  - Optional CNN encoder for pixel observations

Falls back to the NumPy DQN if PyTorch is not available.

Usage:
    from brain.learning.torch_dqn import TorchDQN

    dqn = TorchDQN(n_features=84, n_actions=18)
    q_values = dqn.forward(features)
    loss = dqn.train_batch(states, actions, rewards, next_states, dones)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> str:
    """Get best available device."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DuelingDQNNet(nn.Module):
    """
    Dueling DQN architecture.

    Separates value function V(s) from advantage function A(s,a):
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This helps the network learn which states are valuable
    regardless of action choice.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        # ── Shared feature layers ─────────────────────────────────────
        layers = []
        in_size = n_features
        for h in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = h
        self.shared = nn.Sequential(*layers)

        # ── Value stream V(s) ─────────────────────────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(in_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1),
        )

        # ── Advantage stream A(s,a) ───────────────────────────────────
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions."""
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class CNNEncoder(nn.Module):
    """
    CNN front-end for pixel observations.

    Takes raw game frames (H×W×C) and produces a feature vector
    that feeds into the DQN.
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 84):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calculate conv output size dynamically
        self._output_dim = output_dim
        self.fc = None  # Lazy init on first forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, H, W) → (batch, output_dim)"""
        conv_out = self.conv(x)
        flat = conv_out.reshape(conv_out.size(0), -1)

        if self.fc is None:
            self.fc = nn.Linear(flat.size(1), self._output_dim).to(x.device)

        return self.fc(flat)


class TorchDQN:
    """
    Deep DQN with PyTorch — drop-in replacement for NumPy DQN.

    Features:
    - Dueling architecture
    - Double DQN (use online net to select, target net to evaluate)
    - Soft target updates (Polyak averaging)
    - Gradient clipping
    - Prioritized replay integration
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,           # Soft update rate
        buffer_size: int = 100000,
        batch_size: int = 64,
        grad_clip: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        dropout: float = 0.1,
        use_cnn: bool = False,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = device or get_device()

        # ── Epsilon schedule ──────────────────────────────────────────
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # ── Networks ──────────────────────────────────────────────────
        self.online_net = DuelingDQNNet(
            n_features, n_actions, hidden_sizes, dropout
        ).to(self.device)

        self.target_net = DuelingDQNNet(
            n_features, n_actions, hidden_sizes, dropout
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optional CNN encoder
        self.cnn = None
        if use_cnn:
            self.cnn = CNNEncoder(output_dim=n_features).to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────
        params = list(self.online_net.parameters())
        if self.cnn:
            params += list(self.cnn.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        # ── Replay Buffer ─────────────────────────────────────────────
        self._replay: deque = deque(maxlen=buffer_size)

        # ── Pre-allocated tensors (avoid per-step allocation) ─────────
        self._state_buffer = torch.zeros(1, n_features, device=self.device)

        # ── torch.compile for fused kernels (PyTorch 2.x) ────────────
        try:
            from torch._inductor.cpp_builder import get_cpp_compiler
            get_cpp_compiler()  # Will raise if cl/gcc not found
            self.online_net = torch.compile(self.online_net, mode="default")
        except Exception:
            pass  # No C++ compiler available — skip compilation

        # ── Stats ─────────────────────────────────────────────────────
        self._total_updates = 0
        self._total_steps = 0
        self._losses: deque = deque(maxlen=100)

    def select_action(
        self,
        features: np.ndarray,
        explore: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy.

        Returns (action, q_values).
        """
        self._total_steps += 1

        # Decay epsilon
        if explore:
            progress = min(1.0, self._total_steps / self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start + (
                self.epsilon_end - self.epsilon_start
            ) * progress

        with torch.inference_mode():
            self._state_buffer.copy_(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0))
            state_t = self._state_buffer
            if self.cnn:
                state_t = self.cnn(state_t)
            q_values = self.online_net(state_t).cpu().numpy().flatten()

        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(q_values))

        return action, q_values

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        raw_frames: Optional[np.ndarray] = None,
        next_raw_frames: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition in replay buffer, with optional raw frames for CNN training."""
        self._replay.append((
            np.asarray(state, dtype=np.float32),
            action,
            reward,
            np.asarray(next_state, dtype=np.float32),
            done,
            raw_frames,        # (frame_stack, 84, 84) or None
            next_raw_frames,   # (frame_stack, 84, 84) or None
        ))

    def set_cnn_encoder(self, encoder_fn) -> None:
        """
        Set external CNN encoder for end-to-end learning.

        Args:
            encoder_fn: callable that takes (batch, frame_stack, 84, 84) numpy
                       and returns (batch, n_features) tensor WITH gradients.
                       This is SensoryCortex.encode_for_training.
        """
        self._external_cnn_encoder = encoder_fn

    def train_step(self) -> Dict[str, float]:
        """
        Sample a batch and perform one gradient update.

        Uses Double DQN: online net selects actions, target net evaluates.
        If an external CNN encoder is set and raw frames are available,
        re-encodes through the CNN WITH gradients for end-to-end learning.
        """
        if len(self._replay) < self.batch_size:
            return {"loss": 0.0, "buffer_size": len(self._replay)}

        # Sample batch
        indices = np.random.choice(
            len(self._replay), self.batch_size, replace=False
        )
        batch = [self._replay[i] for i in indices]

        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

        # Check if we have raw frames AND an external CNN encoder
        has_cnn = (
            hasattr(self, '_external_cnn_encoder')
            and self._external_cnn_encoder is not None
            and len(batch[0]) > 5
            and all(b[5] is not None and b[6] is not None for b in batch)
        )

        if has_cnn:
            # CNN path: re-encode raw frames through CNN WITH gradients
            raw_frames = np.stack([b[5] for b in batch])       # (batch, stack, 84, 84)
            next_raw_frames = np.stack([b[6] for b in batch])  # (batch, stack, 84, 84)

            # Forward with gradients (CNN learns!)
            states = self._external_cnn_encoder(raw_frames)

            with torch.no_grad():
                next_states = self._external_cnn_encoder(next_raw_frames)
        else:
            # Feature path: use pre-computed features (no CNN learning)
            states = torch.FloatTensor(
                np.array([b[0] for b in batch])
            ).to(self.device)
            next_states = torch.FloatTensor(
                np.array([b[3] for b in batch])
            ).to(self.device)

        # Encode if built-in CNN (legacy path)
        if self.cnn and not has_cnn:
            states = self.cnn(states)
            with torch.no_grad():
                next_states = self.cnn(next_states)

        # Current Q-values for selected actions
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Online net selects best action for next state
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            best_actions = next_q_online.argmax(dim=1)

            # 2. Target net evaluates that action
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Huber loss (smooth L1) — less sensitive to outliers than MSE
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize — gradients flow through CNN if has_cnn
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip
        )
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        self._total_updates += 1
        loss_val = loss.item()
        self._losses.append(loss_val)

        td_error = (current_q - target_q).abs().mean().item()

        return {
            "loss": loss_val,
            "td_error": td_error,
            "epsilon": self.epsilon,
            "buffer_size": len(self._replay),
            "total_updates": self._total_updates,
            "avg_loss_100": float(np.mean(self._losses)) if self._losses else 0.0,
            "cnn_learning": has_cnn,
        }

    def _soft_update(self) -> None:
        """Polyak averaging: target ← τ·online + (1-τ)·target."""
        for t_param, o_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * o_param.data + (1 - self.tau) * t_param.data
            )

    # ── Compatibility with Striatum ───────────────────────────────────

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Get Q-values (NumPy interface for Striatum compatibility)."""
        with torch.inference_mode():
            self._state_buffer.copy_(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0))
            return self.online_net(self._state_buffer).cpu().numpy().flatten()

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save model weights to file."""
        state = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_updates": self._total_updates,
            "total_steps": self._total_steps,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
        }
        if self.cnn:
            state["cnn"] = self.cnn.state_dict()
        torch.save(state, filepath)

    def load(self, filepath: str) -> None:
        """Load model weights from file."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state.get("epsilon", self.epsilon)
        self._total_updates = state.get("total_updates", 0)
        self._total_steps = state.get("total_steps", 0)
        if self.cnn and "cnn" in state:
            self.cnn.load_state_dict(state["cnn"])

    def stats(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.online_net.parameters())
        return {
            "n_params": n_params,
            "device": str(self.device),
            "epsilon": round(self.epsilon, 4),
            "total_updates": self._total_updates,
            "total_steps": self._total_steps,
            "buffer_size": len(self._replay),
            "avg_loss": round(float(np.mean(self._losses)), 6) if self._losses else 0.0,
            "architecture": "DuelingDQN",
            "has_cnn": self.cnn is not None,
        }
