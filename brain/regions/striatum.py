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
from brain.learning.option_critic import OptionCritic

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

        # ── PyTorch Deep DQN (optional) ────────────────────────────────────
        # Tries RainbowDQN first (NoisyNets+IS-weights+Double+Dueling)
        # Falls back to TorchDQN, then NumPy DQN.
        self._torch_dqn = None
        if use_torch:
            try:
                from brain.learning.torch_dqn import RainbowDQN
                # Adaptive NoisyNet std_init: 0.17 baseline for 4-dim envs,
                # scales up with sqrt(n_features/4) so larger obs spaces get
                # proportionally louder noise (Acrobot=0.21, MWM=0.24, etc.)
                _std_init = float(0.17 * (n_features / 4.0) ** 0.5)
                self._torch_dqn = RainbowDQN(
                    n_features=n_features,
                    n_actions=n_actions,
                    hidden_sizes=(256, 256, 128),
                    lr=lr,
                    gamma=gamma,
                    batch_size=batch_size,
                    std_init=_std_init,
                )
            except Exception:
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
        self._n_steps = 4  # Keep in sync with NStepBuffer below
        self._nstep: NStepBuffer = NStepBuffer(n=self._n_steps, gamma=gamma, downstream=self._replay)
        # Precompute gamma^n for n-step Bellman target
        self._gamma = gamma
        self._gamma_n = gamma ** self._n_steps  # e.g. 0.99^4 ≈ 0.9606

        # ── State ─────────────────────────────────────────────────────
        self._epsilon = 0.15  # Default (EXECUTE mode)
        self._action_bias: Optional[np.ndarray] = None  # From prefrontal strategy
        self._total_updates = 0
        self._episode_reward = 0.0
        self._episode_rewards: deque = deque(maxlen=100)
        self._msg_poll_counter = 0  # Throttle message polling to every 5 steps
        self._episode_count = 0     # For elite fraction schedule
        self._elite_buf = None      # Set by set_elite_buffer() after init

        # Phase 6 Fix 1: track episode step counts for OC warmup auto-scale
        self._episode_step_counts: deque = deque(maxlen=50)
        self._current_ep_steps: int = 0

        # Phase 6 Fix 3: SR reward shaping module (set by set_sr_module())
        self._sr_matrix = None

        # Pre-allocated buffers for zero-alloc forward pass
        self._h_buf = np.zeros(hidden_size, dtype=np.float32)  # reusable hidden layer

        # ── Option-Critic (Phase 4) ──────────────────────────────────
        # Disabled by default. Enabled by orchestrator via enable_option_critic().
        # When ready (>=500 updates), replaces ε-greedy with option-guided selection.
        self._option_critic: Optional[OptionCritic] = None
        self._oc_last_state: Optional[np.ndarray] = None  # for update() call in learn()
        self._oc_last_option: Optional[int] = None
        self._oc_last_action: Optional[int] = None

        # ── Phase 5 Part 3: Integrated Context (Past+Present+Future) ────────
        # When set, OptionCritic.select_option() receives a wider ctx vector:
        #   ctx = concat([elite_embedding, features, dream_features])
        # Rebuilt every 4 steps (oc_ctx_interval) to amortize dreamer cost.
        self._ctx_elite_buf = None    # EliteReplayBuffer ref
        self._ctx_dreamer = None      # DreamerWorldModel ref
        self._ctx_cache: Optional[np.ndarray] = None  # most recent ctx vector
        self._ctx_step: int = 0
        self._oc_ctx_interval: int = 4  # rebuild ctx every N steps

    # ── Elite Buffer wiring ───────────────────────────────────────────

    def set_elite_buffer(self, elite) -> None:
        """
        Wire an EliteReplayBuffer from the Hippocampus.

        Called once by the orchestrator after both regions are created.
        Once set, the Striatum will blend elite transitions into every
        training batch at the decaying fraction elite.elite_fraction(ep).
        """
        self._elite_buf = elite

    def enable_option_critic(self, n_options: int = 4) -> None:
        """
        Activate Option-Critic for this Striatum instance.

        Fix 1: auto-scales min_updates based on recent mean episode length.
          min_updates = max(100, 3 × avg_ep_steps)
        For Acrobot (avg ~100 steps): min_updates=300 → ~3 eps warmup (not 5).
        For Taxi (avg ~200 steps):    min_updates=600 → ~3 eps warmup.

        Falls back silently on import failure.
        """
        try:
            # Estimate avg episode length from recent episode rewards timing.
            # _episode_rewards stores per-episode total rewards; we use length of
            # _nstep buffer's flush history as a proxy for step counts.
            # Simpler: use n_features as a rough env-complexity heuristic,
            # but BEST is the actual step count if available.
            avg_steps = 200  # conservative default
            if hasattr(self, '_episode_step_counts') and len(self._episode_step_counts) > 0:
                avg_steps = int(np.mean(list(self._episode_step_counts)[-20:]))

            warmup = max(100, 3 * avg_steps)

            self._option_critic = OptionCritic(
                n_options=n_options,
                n_actions=self.n_actions,
                n_features=self.n_features,
                gamma=self._gamma,
                min_updates=warmup,
            )
        except Exception:
            self._option_critic = None

    def set_sr_module(self, sr_matrix) -> None:
        """
        Wire a SRMatrix for Fix 3 (SR reward shaping).

        When set, learn() adds an intrinsic shaping bonus:
          r_shaped = r + 0.1 * (sr_dist_before - sr_dist_after)
        This provides a dense pull toward subgoals in sparse-reward envs.
        """
        self._sr_matrix = sr_matrix

    def set_context_sources(
        self,
        elite_buf,
        dreamer,
        elite_embed_dim: int = 8,
        dream_dim: int = 8,
    ) -> None:
        """
        Wire past+future context sources for integrated prediction input (Part 3).

        When set, OptionCritic's Manager receives:
          ctx = concat([elite_embedding(elite_embed_dim), features, dream_features(dream_dim)])

        Calls set_context_mode() on the OptionCritic to expand its weight
        matrices to ctx_dim = elite_embed_dim + n_features + dream_dim.

        Args:
            elite_buf: EliteReplayBuffer with .summary_embedding(dim) method.
            dreamer:   DreamerWorldModel with .dream_latent(features) method.
            elite_embed_dim: dimension of the elite embedding (default 8).
            dream_dim: dimension used from the dream latent (default 8).
        """
        self._ctx_elite_buf = elite_buf
        self._ctx_dreamer = dreamer
        self._ctx_embed_dim = elite_embed_dim
        self._ctx_dream_dim = dream_dim

        if self._option_critic is not None:
            ctx_dim = elite_embed_dim + self.n_features + dream_dim
            try:
                self._option_critic.set_context_mode(ctx_dim)
            except Exception:
                pass

    def _build_ctx(self, features: np.ndarray) -> np.ndarray:
        """
        Build the integrated context vector: [elite_embedding | features | dream_features].

        Falls back gracefully to features-only if sources aren't set or fail.
        Rebuilds every _oc_ctx_interval steps; caches between rebuilds.
        """
        self._ctx_step += 1

        # Cache: only rebuild ctx every 4 steps
        if self._ctx_cache is not None and self._ctx_step % self._oc_ctx_interval != 0:
            # Splice in the fresh current features (position elite_embed_dim)
            ed = getattr(self, "_ctx_embed_dim", 0)
            self._ctx_cache[ed: ed + self.n_features] = features
            return self._ctx_cache

        parts = []
        # 1. Past: elite buffer summary embedding
        try:
            ed = getattr(self, "_ctx_embed_dim", 8)
            if self._ctx_elite_buf is not None and hasattr(self._ctx_elite_buf, "summary_embedding"):
                elite_emb = self._ctx_elite_buf.summary_embedding(ed)
            else:
                elite_emb = np.zeros(ed, dtype=np.float32)
            parts.append(elite_emb.astype(np.float32))
        except Exception:
            ed = getattr(self, "_ctx_embed_dim", 8)
            parts.append(np.zeros(ed, dtype=np.float32))

        # 2. Present: current features
        parts.append(features.astype(np.float32))

        # 3. Future: dreamer latent rollout
        try:
            dd = getattr(self, "_ctx_dream_dim", 8)
            if self._ctx_dreamer is not None and hasattr(self._ctx_dreamer, "dream_latent"):
                dream = np.asarray(self._ctx_dreamer.dream_latent(features), dtype=np.float32)
                dream = dream[:dd] if len(dream) >= dd else np.pad(dream, (0, dd - len(dream)))
            else:
                dream = np.zeros(dd, dtype=np.float32)
            parts.append(dream)
        except Exception:
            dd = getattr(self, "_ctx_dream_dim", 8)
            parts.append(np.zeros(dd, dtype=np.float32))

        ctx = np.concatenate(parts, axis=0)
        self._ctx_cache = ctx
        return ctx

    def oc_observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Feed a transition into OptionCritic for training.
        Called every step by the orchestrator (or learn() path).
        NO-OP if OC not enabled.
        """
        if self._option_critic is None:
            return
        option = self._oc_last_option
        if option is None:
            return
        try:
            self._option_critic.update(state, option, action, reward, next_state, done)
            # Check termination at next state
            self._option_critic.should_terminate(next_state, option)
        except Exception:
            pass

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

        # ── Option-Critic override with integrated ctx (Phase 4+5) ───────
        # When ready, use ctx = concat([elite, features, dream]) for option selection.
        if self._option_critic is not None and self._option_critic.is_ready and explore:
            oc = self._option_critic
            # Build context: features-only when no ctx sources set, else full ctx
            if self._ctx_elite_buf is not None or self._ctx_dreamer is not None:
                oc_input = self._build_ctx(features_arr)
            else:
                oc_input = features_arr
            # Select option if none active
            if oc.current_option is None:
                option = oc.select_option(oc_input, explore=True)
            else:
                option = oc.current_option
            action = oc.intra_option_action(features_arr, option, explore=True)
            self._oc_last_state = features_arr.copy()
            self._oc_last_option = option
            self._oc_last_action = action
            # Still compute Q-values for logging (non-blocking, torch path)
            if self._torch_dqn is not None:
                _, q_values = self._torch_dqn.select_action(features_arr, explore=False)
            else:
                q_values = self._forward(features_arr)
            return {
                "action": action,
                "q_values": q_values,
                "epsilon": 0.0,
                "backend": "option_critic",
                "option": option,
            }

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
            # Fix 1: track step counts for OC warmup auto-scale
            self._current_ep_steps += 1
            self._episode_reward += reward
            if done:
                self._episode_rewards.append(self._episode_reward)
                self._episode_step_counts.append(self._current_ep_steps)
                self._episode_reward = 0.0
                self._current_ep_steps = 0

            # Throttled: skip gradient update if requested
            if experience.get("skip_train", False):
                return {"loss": 0.0, "backend": "torch", "buffer_size": len(self._torch_dqn._replay)}

            train_out = self._torch_dqn.train_step()
            # RainbowDQN returns 3-tuple (stats, td_errors, self_surprises)
            # TorchDQN returns 2-tuple (stats, td_errors) — handle both
            if isinstance(train_out, tuple) and len(train_out) == 3:
                result, _td, _self_surp = train_out
            elif isinstance(train_out, tuple) and len(train_out) == 2:
                result, _td = train_out
                _self_surp = None
            else:
                result = train_out
                _self_surp = None

            result["backend"] = "torch"
            return result

        # ── NumPy fallback ────────────────────────────────────────────
        # Fix 3: SR reward shaping (dense pull toward subgoals)
        if self._sr_matrix is not None:
            try:
                sr_ready = getattr(self._sr_matrix, 'is_ready', False)
                if sr_ready:
                    d_before = float(self._sr_matrix.distance(state_arr))
                    d_after  = float(self._sr_matrix.distance(next_state_arr))
                    reward   = reward + 0.1 * (d_before - d_after)
            except Exception:
                pass

        # Push via n-step buffer → StratifiedReplayDeque (PER)
        near_death = bool(reward < -0.5 or (done and reward <= 0))
        self._nstep.push(state_arr, action, reward, next_state_arr, done,
                         near_death=near_death)

        # Fix 1: track episode step counts
        self._current_ep_steps += 1
        self._episode_reward += reward
        if done:
            self._episode_rewards.append(self._episode_reward)
            self._episode_step_counts.append(self._current_ep_steps)
            self._episode_reward = 0.0
            self._current_ep_steps = 0
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
        # MAJOR-9 fix: NStepBuffer flushes G_t (sum of n discounted rewards) + s_n.
        # The remaining bootstrap must be discounted by gamma^n, not gamma^1.
        # Old: gamma * max Q(s_n) — systematic underestimation by (gamma^n - gamma).
        q_target = rewards + self._gamma_n * np.max(q_next, axis=1) * (1 - dones)

        # TD error
        td_error = q_target - q_selected
        loss = float(np.mean(td_error ** 2))

        # Backward: gradient update
        self._backward_dynamic(states, actions, td_error, actual_bs)

        # Fix 2: PER priority = TD error + 0.3 * self_surprise (NumPy path)
        # self_surprise measures behavioral inconsistency from the self-model head.
        # In NumPy path there's no self-model head, so use |td_error| as priority.
        if len(sampled_indices) > 0:
            per_priority = np.abs(td_error[:len(sampled_indices)])
            self._replay.update_priorities(sampled_indices, per_priority)

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
        """Backward pass supporting variable batch sizes.

        Includes SAC-style entropy regularization (Christodoulou 2019):
        entropy bonus keeps Q-values spread across actions, preventing
        premature exploitation collapse even in Rainbow/noisy-net mode.
        """
        hidden = np.maximum(0, states @ self._W1 + self._b1)

        # dL/dQ for selected actions
        dQ = np.zeros((batch_size, self.n_actions))
        dQ[np.arange(batch_size), actions] = -2 * td_error / batch_size

        # SAC entropy bonus: -alpha * sum(pi * log(pi))
        # nudges Q-values to stay spread; alpha=0.01 is mild
        q_all = self._forward_batch(states)                         # (bs, n_actions)
        pi = np.exp(q_all - q_all.max(axis=1, keepdims=True))      # softmax numerically stable
        pi /= pi.sum(axis=1, keepdims=True) + 1e-8
        entropy_grad = -0.01 * (1.0 + np.log(pi + 1e-8)) / batch_size   # d(-H)/dQ
        dQ += entropy_grad

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
