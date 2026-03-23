"""
striatum.py — Action Selection & Policy Learning Region.

Responsible for:
  - Action-value learning (DQN, PPO, or other RL algorithm)
  - Policy execution with epsilon-greedy exploration
  - Receiving threat assessment from AmygdalaThalamus to adjust epsilon
  - Receiving strategy from PrefrontalCortex to bias action selection

This is where the unified learning pipeline lives, resolving the
throng3/4 redundancy between PortableNNAgent and MetaStackPipeline.
The Striatum uses a single configurable learner (defaulting to DQN).

In Phase 3, the learner becomes swappable via RLZoo integration.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion
from brain.learning.replay_adapters import StratifiedReplayDeque, NStepBuffer

# ── Optional Numba JIT for hot forward pass ────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def _njit(fn=None, **kw):           # noqa: E302 — passthrough decorator
        if fn is not None: return fn
        def _inner(f): return f
        return _inner


@_njit(cache=True)
def _dqn_forward_nb(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """
    Two-layer ReLU DQN forward pass.
    When compile with Numba this runs at C speed (~10-50x vs pure numpy).
    When Numba is absent, falls back to equivalent numpy.
    """
    hidden = x @ W1 + b1
    # ReLU
    for i in range(len(hidden)):
        if hidden[i] < 0.0:
            hidden[i] = 0.0
    return hidden @ W2 + b2


class Striatum(BrainRegion):
    """
    Action-value learning and policy execution region.

    Unified learner replacing the dual PortableNNAgent/MetaStackPipeline paths.
    Default: lightweight DQN with configurable architecture.

    Listens to:
      - AmygdalaThalamus → adjusts epsilon based on operating mode
      - PrefrontalCortex → receives strategy suggestions (action biases)
      - Hippocampus → receives replay data for offline training
    """

    def __init__(
        self,
        bus: MessageBus,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_size: int = 128,
        gamma: float = 0.99,
        lr: float = 0.001,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        use_torch: bool = False,
    ):
        super().__init__("striatum", bus)

        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ── PyTorch Deep DQN (optional) ───────────────────────────────
        self._torch_dqn = None
        if use_torch:
            try:
                from brain.learning.torch_dqn import TorchDQN
                self._torch_dqn = TorchDQN(
                    n_features=n_features,
                    n_actions=n_actions,
                    hidden_sizes=(256, 256, 128),
                    lr=lr,
                    gamma=gamma,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                )
            except ImportError:
                pass  # Fall back to NumPy DQN

        # ── DQN Network (online) ──────────────────────────────────────
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / hidden_size)

        self._W1 = rng.randn(n_features, hidden_size).astype(np.float32) * scale1
        self._b1 = np.zeros(hidden_size, dtype=np.float32)
        self._W2 = rng.randn(hidden_size, n_actions).astype(np.float32) * scale2
        self._b2 = np.zeros(n_actions, dtype=np.float32)

        # ── DQN Network (target — frozen copy) ───────────────────────
        self._tW1 = self._W1.copy()
        self._tb1 = self._b1.copy()
        self._tW2 = self._W2.copy()
        self._tb2 = self._b2.copy()

        # ── Replay Buffer (PER-stratified) ──────────────────────────────
        # StratifiedReplayDeque: 30% near-death / 70% normal, TD-error weighted
        self._replay: StratifiedReplayDeque = StratifiedReplayDeque(capacity=buffer_size)
        # NStepBuffer: accumulates n=4 steps, flushes G_t discounted returns
        self._nstep: NStepBuffer = NStepBuffer(n=4, gamma=gamma, downstream=self._replay)

        # ── State ─────────────────────────────────────────────────────
        self._epsilon = 0.15  # Default (EXECUTE mode)
        self._action_bias: Optional[np.ndarray] = None  # From prefrontal strategy
        self._total_updates = 0
        self._episode_reward = 0.0
        self._episode_rewards: deque = deque(maxlen=100)
        self._msg_poll_counter = 0  # Throttle message polling to every 5 steps
        self._episode_count = 0     # For elite fraction schedule
        self._elite_buf = None      # Set by set_elite_buffer() after init

        # Pre-allocated buffers for zero-alloc forward pass
        self._h_buf = np.zeros(hidden_size, dtype=np.float32)  # reusable hidden layer

    # ── Elite Buffer wiring ───────────────────────────────────────────

    def set_elite_buffer(self, elite) -> None:
        """
        Wire an EliteReplayBuffer from the Hippocampus.

        Called once by the orchestrator after both regions are created.
        Once set, the Striatum will blend elite transitions into every
        training batch at the decaying fraction elite.elite_fraction(ep).
        """
        self._elite_buf = elite

    # ── CNN Integration ───────────────────────────────────────────────

    def wire_cnn_encoder(self, encoder_fn, cnn_params) -> None:
        """
        Wire an external CNN encoder for end-to-end learning.

        Args:
            encoder_fn: SensoryCortex.encode_for_training — takes (batch, stack, 84, 84)
                       numpy, returns (batch, n_features) tensor WITH gradients
            cnn_params: list of CNN parameters to add to optimizer
        """
        if self._torch_dqn is not None:
            self._torch_dqn.set_cnn_encoder(encoder_fn)
            # Add CNN params to optimizer
            for param_group in self._torch_dqn.optimizer.param_groups:
                param_group['params'].extend(cnn_params)

    # ── BrainRegion Interface ─────────────────────────────────────────

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select action given current features.

        Expected inputs:
            features: np.ndarray — 84-dim abstract features
            valid_actions: Optional[List[int]] — mask of allowed actions
            explore: bool — whether to explore (default True)
        """
        features = inputs.get("features")
        valid_actions = inputs.get("valid_actions")
        explore = inputs.get("explore", True)

        if features is None:
            return {"action": 0, "q_values": np.zeros(self.n_actions)}

        # Throttle message polling: strategy/threat updates don't need per-step precision
        self._msg_poll_counter += 1
        if self._msg_poll_counter >= 5:
            self._msg_poll_counter = 0
            self._process_messages()

        features_arr = np.asarray(features, dtype=np.float32)

        # ── Use TorchDQN if available ─────────────────────────────────
        if self._torch_dqn is not None:
            action, q_values = self._torch_dqn.select_action(features_arr, explore=explore)

            # Apply action bias from prefrontal cortex
            if self._action_bias is not None:
                q_values = q_values + self._action_bias
                if not explore or np.random.random() >= self._torch_dqn.epsilon:
                    action = int(np.argmax(q_values))

            return {
                "action": action,
                "q_values": q_values,
                "epsilon": self._torch_dqn.epsilon,
                "backend": "torch",
            }

        # ── NumPy fallback ────────────────────────────────────────────
        # Get Q-values
        q_values = self._forward(features_arr)

        # Apply action bias from prefrontal cortex
        if self._action_bias is not None:
            q_values = q_values + self._action_bias

        # Mask invalid actions
        if valid_actions is not None:
            mask = np.full(self.n_actions, -1e9)
            for a in valid_actions:
                mask[a] = 0.0
            q_values = q_values + mask

        # Epsilon-greedy selection
        if explore and np.random.random() < self._epsilon:
            if valid_actions:
                action = int(np.random.choice(valid_actions))
            else:
                action = int(np.random.randint(self.n_actions))
        else:
            action = int(np.argmax(q_values))

        return {
            "action": action,
            "q_values": q_values,
            "epsilon": self._epsilon,
        }

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Learn from a transition or a batch of replay data.

        Expected experience:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
            raw_frames: optional np.ndarray (frame_stack, 84, 84)
            next_raw_frames: optional np.ndarray (frame_stack, 84, 84)
        """
        state = experience.get("state")
        action = experience.get("action")
        reward = experience.get("reward")
        next_state = experience.get("next_state")
        done = experience.get("done")
        raw_frames = experience.get("raw_frames")
        next_raw_frames = experience.get("next_raw_frames")

        if state is None:
            return {"loss": 0.0}

        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)

        # ── Use TorchDQN if available ─────────────────────────────────
        if self._torch_dqn is not None:
            self._torch_dqn.store_transition(
                state_arr, action, reward, next_state_arr, done,
                raw_frames=raw_frames,
                next_raw_frames=next_raw_frames,
            )
            self._episode_reward += reward
            if done:
                self._episode_rewards.append(self._episode_reward)
                self._episode_reward = 0.0

            # Throttled: skip gradient update if requested
            if experience.get("skip_train", False):
                return {"loss": 0.0, "backend": "torch", "buffer_size": len(self._torch_dqn._replay)}

            result = self._torch_dqn.train_step()
            result["backend"] = "torch"
            return result

        # ── NumPy fallback ────────────────────────────────────────────
        # Push via n-step buffer → StratifiedReplayDeque (PER)
        near_death = bool(reward < -0.5 or (done and reward <= 0))
        self._nstep.push(state_arr, action, reward, next_state_arr, done,
                         near_death=near_death)

        self._episode_reward += reward
        if done:
            self._episode_rewards.append(self._episode_reward)
            self._episode_reward = 0.0
            self._episode_count += 1

        # Batch learning from replay
        if len(self._replay) < self.batch_size:
            return {"loss": 0.0, "buffer_size": len(self._replay)}

        # ── PER-stratified batch ──────────────────────────────────
        elite_transitions = []
        if self._elite_buf is not None and not self._elite_buf.is_empty:
            frac = self._elite_buf.elite_fraction(self._episode_count)
            elite_n = max(0, round(self.batch_size * frac))
            if elite_n > 0:
                elite_transitions = self._elite_buf.sample(elite_n)
        normal_n = self.batch_size - len(elite_transitions)

        # Normal replay: PER-stratified sample
        normal_transitions, sampled_indices = self._replay.sample(min(normal_n, len(self._replay)))
        batch = normal_transitions + elite_transitions

        if not batch:
            return {"loss": 0.0, "buffer_size": len(self._replay)}
        actual_bs = len(batch)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Forward: online Q-values
        q_values = self._forward_batch(states)
        q_selected = q_values[np.arange(actual_bs), actions]

        # Forward: target Q-values
        q_next = self._forward_target_batch(next_states)
        q_target = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # TD error
        td_error = q_target - q_selected
        loss = float(np.mean(td_error ** 2))

        # Backward: gradient update
        self._backward_dynamic(states, actions, td_error, actual_bs)

        # PER: update priorities for the normal (non-elite) transitions
        if len(sampled_indices) > 0:
            self._replay.update_priorities(sampled_indices, td_error[:len(sampled_indices)])

        self._total_updates += 1
        if self._total_updates % self.target_update_freq == 0:
            self._sync_target()

        elite_frac = getattr(self._elite_buf, 'elite_fraction', lambda _: 0.0)(self._episode_count) if self._elite_buf else 0.0
        return {
            "loss": loss,
            "td_error_mean": float(np.mean(np.abs(td_error))),
            "buffer_size": len(self._replay),
            "total_updates": self._total_updates,
            "elite_in_batch": len(elite_transitions),
            "elite_fraction": round(elite_frac, 3),
        }

    def report(self) -> Dict[str, Any]:
        base = super().report()
        avg_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        result = {
            **base,
            "epsilon": self._epsilon,
            "buffer_size": len(self._replay),
            "total_updates": self._total_updates,
            "avg_reward_100ep": round(avg_reward, 2),
            "has_action_bias": self._action_bias is not None,
            "backend": "torch" if self._torch_dqn else "numpy",
        }
        if self._torch_dqn:
            result.update(self._torch_dqn.stats())
        return result

    # ── Internal: DQN ─────────────────────────────────────────────────

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """One-sample forward pass. Uses Numba JIT when available."""
        if _NUMBA:
            return _dqn_forward_nb(x, self._W1, self._b1, self._W2, self._b2)
        # Numpy fallback with pre-allocated buffer
        np.dot(x, self._W1, out=self._h_buf)
        self._h_buf += self._b1
        np.maximum(self._h_buf, 0, out=self._h_buf)
        return self._h_buf @ self._W2 + self._b2

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, X @ self._W1 + self._b1)
        return hidden @ self._W2 + self._b2

    def _forward_target_batch(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, X @ self._tW1 + self._tb1)
        return hidden @ self._tW2 + self._tb2

    def _backward(self, states: np.ndarray, actions: np.ndarray, td_error: np.ndarray) -> None:
        """Single backward pass for DQN update (fixed batch_size)."""
        self._backward_dynamic(states, actions, td_error, self.batch_size)

    def _backward_dynamic(self, states: np.ndarray, actions: np.ndarray,
                          td_error: np.ndarray, batch_size: int) -> None:
        """Backward pass supporting variable batch sizes (elite + normal blending)."""
        hidden = np.maximum(0, states @ self._W1 + self._b1)

        # dL/dQ for selected actions
        dQ = np.zeros((batch_size, self.n_actions))
        dQ[np.arange(batch_size), actions] = -2 * td_error / batch_size

        # Layer 2 gradients
        dW2 = hidden.T @ dQ
        db2 = np.sum(dQ, axis=0)

        # Layer 1 gradients
        dhidden = dQ @ self._W2.T
        dhidden[hidden <= 0] = 0  # ReLU
        dW1 = states.T @ dhidden
        db1 = np.sum(dhidden, axis=0)

        # Update
        self._W1 -= self.lr * dW1
        self._b1 -= self.lr * db1
        self._W2 -= self.lr * dW2
        self._b2 -= self.lr * db2

    def _sync_target(self) -> None:
        self._tW1 = self._W1.copy()
        self._tb1 = self._b1.copy()
        self._tW2 = self._W2.copy()
        self._tb2 = self._b2.copy()

    # ── Message processing ────────────────────────────────────────────

    def _process_messages(self) -> None:
        """Process incoming messages from other regions."""
        messages = self.receive(max_messages=5)
        for msg in messages:
            if msg.msg_type == "threat_assessment":
                # Adjust epsilon based on operating mode
                self._epsilon = msg.payload.get("epsilon", self._epsilon)
            elif msg.msg_type == "strategy":
                # Apply action bias from prefrontal cortex
                bias = msg.payload.get("action_bias")
                if bias is not None:
                    self._action_bias = np.asarray(bias, dtype=np.float32)
            # replay_batch msgs from hippocampus are unused here;
            # Striatum manages its own _replay deque directly.

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._action_bias = None
        self._episode_reward = 0.0

    def save_weights(self) -> Dict[str, np.ndarray]:
        return {"W1": self._W1, "b1": self._b1, "W2": self._W2, "b2": self._b2}

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self._W1 = weights["W1"]
        self._b1 = weights["b1"]
        self._W2 = weights["W2"]
        self._b2 = weights["b2"]
        self._sync_target()
