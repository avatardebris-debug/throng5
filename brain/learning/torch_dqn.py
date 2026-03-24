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


# ── SPR: Self-Predictive Representations (Schwarzer et al 2021) ───────────


class SPRHead(nn.Module):
    """
    Projection head: maps encoder output → latent representation for SPR loss.

    Two-layer MLP: h(s) → z ∈ R^latent_dim  (L2 normalized).
    Attached to the online encoder; a stop-gradient copy used for targets.

    SPR prevents representation collapse by enforcing that:
        predictor(z_t, a_t) ≈ stop_grad(z_{t+1})
    using cosine similarity as the loss (no contrastive negatives needed).
    """

    def __init__(self, in_dim: int, latent_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, p=2, dim=-1)   # L2 norm → unit sphere


class SPRTransitionPredictor(nn.Module):
    """
    Latent transition predictor: (z_t, a_t) → z_{t+1}_pred.

    Concatenates current latent z_t with a one-hot action embedding,
    then applies a 2-layer MLP to predict next-state latent.
    """

    def __init__(self, latent_dim: int = 128, n_actions: int = 18):
        super().__init__()
        self.n_actions = n_actions
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self, z: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        z       : (batch, latent_dim)
        actions : (batch,) — integer action indices
        returns : (batch, latent_dim) predicted next latent (normalized)
        """
        a_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        inp = torch.cat([z, a_onehot], dim=-1)
        z_pred = self.predictor(inp)
        return F.normalize(z_pred, p=2, dim=-1)


def spr_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine similarity loss: L = -mean(cosine_sim(z_pred, stop_grad(z_target))).
    Range [-1, 1] → target is 0 (identical) direction.
    """
    return -(z_pred * z_target.detach()).sum(dim=-1).mean()


# ── NoisyLinear (Fortunato et al 2017) ────────────────────────────────────


class NoisyLinear(nn.Module):
    """
    Linear layer with factorized Gaussian noise — replaces ε-greedy exploration.

    Each weight: w = μ_w + σ_w ⊙ ε_w  (factorized noise: ε = ε_p ⊗ ε_q)
    Noise is refreshed every forward pass during training, frozen during eval.

    Advantages over ε-greedy:
    - Per-weight exploration (state-dependent, not uniform)
    - Exploration collapses automatically as training progresses (σ → 0)
    - No separate epsilon schedule needed
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable: mean and std for weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / self.in_features ** 0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.std_init / self.in_features ** 0.5)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.std_init / self.out_features ** 0.5)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Refresh factorized noise — call once per step during training."""
        p = self._scale_noise(self.in_features)
        q = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(q.outer(p))
        self.bias_epsilon.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQNNet(nn.Module):
    """
    Rainbow network: Dueling architecture with NoisyLinear streams.

    Shared feature extractor uses standard Linear layers (fast).
    Value and advantage streams use NoisyLinear (exploration).
    Phase 5: self_head predicts the agent's own next option (egocentric).
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        std_init: float = 0.5,
        n_options: int = 4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_options = n_options

        # Shared (deterministic) feature extractor
        layers = []
        in_size = n_features
        for h in hidden_sizes[:-1]:
            layers.extend([nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU()])
            in_size = h
        self.shared = nn.Sequential(*layers)

        # Noisy value stream V(s)
        self.value_1 = NoisyLinear(in_size, hidden_sizes[-1], std_init)
        self.value_2 = NoisyLinear(hidden_sizes[-1], 1, std_init)

        # Noisy advantage stream A(s,a)
        self.adv_1 = NoisyLinear(in_size, hidden_sizes[-1], std_init)
        self.adv_2 = NoisyLinear(hidden_sizes[-1], n_actions, std_init)

        # ── Phase 5: Egocentric Self-Model Head ───────────────────
        # Predicts the agent's own next option choice.
        # self_head(z_t) ≈ one-hot(option_{t+1})
        # Loss: cross_entropy(self_logits, actual_option)
        # Signal: self_surprise = -log P(actual_option | z_t)
        self.self_head = nn.Sequential(
            NoisyLinear(in_size, 64, std_init),
            nn.ReLU(),
            nn.Linear(64, n_options),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_self_pred: bool = False,
    ):
        """Compute Q-values; optionally also return self-option logits."""
        feat = self.shared(x)
        value = self.value_2(F.relu(self.value_1(feat)))
        adv = self.adv_2(F.relu(self.adv_1(feat)))
        q = value + adv - adv.mean(dim=-1, keepdim=True)
        if return_self_pred:
            self_logits = self.self_head(feat)
            return q, self_logits
        return q

    def reset_noise(self) -> None:
        """Refresh all NoisyLinear noise — call once per training step."""
        for m in [self.value_1, self.value_2, self.adv_1, self.adv_2]:
            m.reset_noise()


class RainbowDQN:
    """
    Rainbow DQN — drop-in replacement for TorchDQN.

    Improvements over TorchDQN (which already has Double DQN + Dueling):
    ✓ Noisy Nets: replaces ε-greedy with learned per-weight noise (state-dependent)
    ✓ IS-weighted Huber loss: respects PER importance-sampling weights
    ~ N-step returns: handled upstream by NStepBuffer (Phase 1B)
    ~ PER: handled upstream by StratifiedReplayDeque (Phase 1A)

    Interface is identical to TorchDQN: same select_action/train_step/save/load.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        lr: float = 6.25e-5,          # Rainbow paper default
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 32,
        grad_clip: float = 10.0,
        std_init: float = 0.17,       # NoisyLinear init std (0.17 for non-Atari; paper used 0.5 for pixel nets)
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

        # Networks
        self.online_net = RainbowDQNNet(
            n_features, n_actions, hidden_sizes, std_init
        ).to(self.device)
        self.target_net = RainbowDQNNet(
            n_features, n_actions, hidden_sizes, std_init
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # NoisyNets → no ε-greedy needed; expose epsilon=0 for compatibility
        self.epsilon = 0.0
        self.cnn = None                    # CNN not used by Rainbow currently

        # Stats
        self._total_updates = 0
        self._total_steps = 0
        self._losses: deque = deque(maxlen=100)

        # Pre-alloc
        self._state_buffer = torch.zeros(1, n_features, device=self.device)

        # ── SPR: Self-Predictive Representations ───────────────────────
        # shared encoder dim = last hidden size before noisy streams
        _enc_dim = hidden_sizes[-2] if len(hidden_sizes) >= 2 else 256
        self.spr_head = SPRHead(in_dim=_enc_dim, latent_dim=128)
        self.spr_head_target = SPRHead(in_dim=_enc_dim, latent_dim=128)
        self.spr_head_target.load_state_dict(self.spr_head.state_dict())
        self.spr_predictor = SPRTransitionPredictor(latent_dim=128, n_actions=n_actions)

        # ── Phase 5: SparseRegrowth (dynamic neurogenesis/pruning) ─────────
        # Operates on shared layers every 500 train steps.
        # Pruning: low-magnitude + low-gradient weights → zeroed (dead synapses)
        # Regrowth: zero weights with strong gradient → random small reinit (birth)
        try:
            from brain.learning.sparse_regrowth import SparseRegrowth
            self._sparse = SparseRegrowth(
                model=self.online_net,
                target_layers=["shared"],
                prune_interval=500,
                max_sparsity=0.40,
            )
        except Exception:
            self._sparse = None

        # ── Phase 5: self_surprise history (ring buffer) ───────────────
        # Stores per-step self_surprise for priority blending in hippocampus
        self._self_surprise_buf: deque = deque(maxlen=1000)


    # ── Action selection ──────────────────────────────────────────────

    def select_action(
        self,
        features: np.ndarray,
        explore: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """Select action using noisy network (no ε-greedy)."""
        self._total_steps += 1
        self.online_net.train(explore)   # Enable noise during exploration
        with torch.no_grad():
            self._state_buffer.copy_(
                torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            )
            q_values = self.online_net(self._state_buffer).cpu().numpy().flatten()
        action = int(np.argmax(q_values))
        return action, q_values

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Q-values (NumPy interface for Striatum compatibility)."""
        with torch.inference_mode():
            self._state_buffer.copy_(
                torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            )
            return self.online_net(self._state_buffer).cpu().numpy().flatten()

    # ── Transition storage (deque — upstream PER/NStep handle priority logic) ──

    def store_transition(self, state, action, reward, next_state, done,
                         raw_frames=None, next_raw_frames=None) -> None:
        """No-op: Striatum._nstep → _replay handle storage."""
        pass

    # ── Training ──────────────────────────────────────────────────────

    def train_step(
        self,
        batch=None,
        is_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        One gradient update.

        If batch is provided (list of (s,a,r,s',done)), uses it directly.
        Otherwise, this is a no-op (Striatum drives batch construction).
        """
        if batch is None or len(batch) == 0:
            return {"loss": 0.0, "backend": "rainbow"}, np.array([])

        n = len(batch)
        states     = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions    = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards    = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones      = torch.FloatTensor([float(t[4]) for t in batch]).to(self.device)

        # IS weights (from PER); ones if not provided
        if is_weights is not None:
            weights = torch.FloatTensor(is_weights[:n]).to(self.device)
        else:
            weights = torch.ones(n, device=self.device)

        # Reset noise before forward (online only — target should be deterministic)
        self.online_net.reset_noise()
        self.online_net.train()

        # Double DQN: online selects, target evaluates
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        td_errors = (current_q - target_q).abs().detach()

        # SPR auxiliary loss: encoder predicts next-state latent
        try:
            shared_feat = self.online_net.shared(states)
            with torch.no_grad():
                next_shared_feat = self.spr_head_target(self.online_net.shared(next_states))
            z_pred = self.spr_predictor(self.spr_head(shared_feat), actions)
            spr_aux = spr_loss(z_pred, next_shared_feat)
        except Exception:
            spr_aux = torch.tensor(0.0, device=self.device)

        # ── Phase 5: Egocentric self-model loss ────────────────────────
        # target_options: the option the agent actually picked at each step.
        # Passed in via batch metadata slot [5] if available.
        self_loss = torch.tensor(0.0, device=self.device)
        batch_self_surprises = np.zeros(n, dtype=np.float32)
        try:
            # Extract option targets from batch metadata (slot 5)
            target_opts = [t[5] if len(t) > 5 else -1 for t in batch]
            valid = [i for i, o in enumerate(target_opts) if o >= 0]
            if valid:
                _, self_logits = self.online_net(states[valid], return_self_pred=True)
                opt_tensor = torch.LongTensor(
                    [target_opts[i] for i in valid]
                ).to(self.device)
                self_loss_vec = F.cross_entropy(
                    self_logits, opt_tensor, reduction="none"
                )
                self_loss = self_loss_vec.mean()
                # Per-transition self-surprise for PER priority boost
                surp = self_loss_vec.detach().cpu().numpy()
                for ii, vi in enumerate(valid):
                    batch_self_surprises[vi] = float(surp[ii])
                self._self_surprise_buf.extend(surp.tolist())
        except Exception:
            pass

        # IS-weighted Huber loss + SPR + self-model
        elementwise = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * elementwise).mean() + 0.1 * spr_aux + 0.05 * self_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network (Q-value)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(self.tau * o_p.data + (1 - self.tau) * t_p.data)

        # EMA update SPR target encoder (CRITICAL-7: without this, SPR loss is noise)
        for t_p, o_p in zip(self.spr_head_target.parameters(), self.spr_head.parameters()):
            t_p.data.copy_(self.tau * o_p.data + (1 - self.tau) * t_p.data)

        self._total_updates += 1
        loss_val = loss.item()
        self._losses.append(loss_val)

        # ── Phase 5: SparseRegrowth cycle (every 500 steps) ──────────────
        if self._sparse is not None:
            self._sparse.maybe_step(self._total_updates)

        return {
            "loss": loss_val,
            "td_error": td_errors.mean().item(),
            "epsilon": 0.0,
            "backend": "rainbow",
            "total_updates": self._total_updates,
            "avg_loss_100": float(np.mean(self._losses)),
            "self_surprise": float(batch_self_surprises.mean()),
            "sparsity": self._sparse._last_sparsity if self._sparse else 0.0,
        }, td_errors.cpu().numpy(), batch_self_surprises

    def self_surprise(self, features: np.ndarray, option: int) -> float:
        """
        Compute egocentric self-surprise at current state.

        Returns -log P(option | z_t): higher = agent would not have
        predicted picking this option. Used for one-step priority query
        without a full batch.
        """
        try:
            s = torch.as_tensor(features, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            self.online_net.eval()
            with torch.no_grad():
                _, logits = self.online_net(s, return_self_pred=True)
                loss = F.cross_entropy(
                    logits,
                    torch.tensor([option], device=self.device),
                )
            return float(loss.item())
        except Exception:
            return 0.0

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self._total_updates,
            "total_steps": self._total_steps,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
        }, filepath)

    def load(self, filepath: str) -> None:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._total_updates = state.get("total_updates", 0)
        self._total_steps = state.get("total_steps", 0)

    def stats(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.online_net.parameters())
        return {
            "n_params": n_params,
            "device": str(self.device),
            "epsilon": 0.0,
            "total_updates": self._total_updates,
            "total_steps": self._total_steps,
            "architecture": "RainbowDQN (NoisyDueling+DoubleDQN)",
        }

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
