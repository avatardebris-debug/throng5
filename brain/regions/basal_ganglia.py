"""
basal_ganglia.py — Procedure Selection & Habit Learning Region.

Wraps:
  - MiniSNN (400-neuron spiking neural network for context matching)
  - WorldModel (learned dynamics model for dream simulation)
  - CompressedState encoder
  - Habit Network (game-specific DQN for pretrained action selection)

Responsible for:
  - Context recognition via SNN (familiar vs novel states)
  - Running dream simulations via world model look-ahead
  - Procedure/habit selection based on learned patterns
  - Feeding dream results to Amygdala and Hippocampus
  - Providing pretrained habit Q-values from game-specific DQN
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion

# Lazy import world model (may need torch)
_WorldModel = None

def _get_world_model_class():
    global _WorldModel
    if _WorldModel is None:
        try:
            from brain.learning.world_model import WorldModel
            _WorldModel = WorldModel
        except ImportError:
            _WorldModel = False
    return _WorldModel if _WorldModel is not False else None


class BasalGanglia(BrainRegion):
    """
    Procedure selection and prediction region.

    Uses SNN for context matching and WorldModel for dream simulations.
    Produces dream results consumed by AmygdalaThalamus and Hippocampus.

    The world model learns to predict (next_state, reward) from (state, action).
    When dreaming, it rolls out imagined trajectories using the learned model,
    then evaluates them to produce action biases.
    """

    def __init__(
        self,
        bus: MessageBus,
        snn_input_dim: int = 16,
        snn_n_neurons: int = 400,
        state_size: int = 64,
        n_features: int = 84,
        n_actions: int = 18,
        dream_interval: int = 5,
        dream_horizon: int = 5,
        use_world_model: bool = True,
    ):
        super().__init__("basal_ganglia", bus)

        self.snn_input_dim = snn_input_dim
        self.snn_n_neurons = snn_n_neurons
        self.state_size = state_size
        self.n_features = n_features
        self.n_actions = n_actions
        self.dream_interval = dream_interval
        self.dream_horizon = dream_horizon

        # SNN for context matching (will be initialized lazily or from weights)
        self._snn_initialized = False
        self._context_score = 0.0

        # ── World Model ───────────────────────────────────────────────
        self._world_model = None
        self._wm_train_steps = 0
        self._wm_confidence = 0.0
        self._policy_fn: Optional[Callable] = None

        if use_world_model:
            WMClass = _get_world_model_class()
            if WMClass is not None:
                self._world_model = WMClass(
                    n_features=n_features,
                    n_actions=n_actions,
                )

        # Dream scheduling
        self._steps_since_dream = 0
        self._total_dreams = 0
        self._last_dream_value = 0.0
        self._last_action_values: Optional[np.ndarray] = None

        # ── Habit Network (game-specific DQN) ─────────────────────────
        # Pretrained on simulator puzzles, provides instant Q-value hints.
        # Loaded via load_habit_weights() after construction.
        self._habit_network = None
        self._habit_encoder = None
        self._habit_confidence = 0.0
        self._habit_weight = 0.5  # Blend weight: habit vs dream action values

        # ── SARSA↔DQN Bridge (dual-process) ──────────────────────────
        # When loaded, supersedes standalone habit network.
        # SARSA handles states DQN is uncertain about; DQN generalizes.
        self._bridge = None

    def set_policy_fn(self, policy_fn: Callable) -> None:
        """
        Set the DQN policy function for dream action selection.

        Args:
            policy_fn: callable(features, explore=False) -> (action, q_values)
        """
        self._policy_fn = policy_fn

    def load_habit_weights(self, weights_path: str,
                           game: str = "lolo") -> bool:
        """
        Load a pretrained game-specific DQN as the habit network.

        The habit network provides instant action Q-values from
        compressed game state — like muscle memory from practice.

        Args:
            weights_path: path to .pt file (from cloud/export_weights.py)
            game: which game the weights are for (determines encoder)

        Returns:
            True if loaded successfully
        """
        try:
            if game == "lolo":
                from brain.games.lolo.lolo_dqn_learner import LoloDQNLearner
                from brain.games.lolo.lolo_compressed_state import LoloCompressedState

                self._habit_network = LoloDQNLearner(n_actions=6)
                self._habit_network.load(weights_path)
                self._habit_network.epsilon = 0.02  # Near-greedy
                self._habit_encoder = LoloCompressedState()
                self._habit_confidence = 1.0

                # Also use the DQN as dream policy
                self._policy_fn = self._habit_policy_fn

                return True
            else:
                return False
        except Exception:
            return False

    def set_habit_simulator(self, sim) -> None:
        """
        Set the simulator reference for the habit network encoder.
        Call this each step with the current game simulator.
        """
        if self._habit_network is not None:
            self._habit_network.set_simulator(sim)

    def _habit_policy_fn(self, features: np.ndarray,
                         explore: bool = False):
        """Policy function wrapper for dream action selection."""
        if self._habit_network is None:
            return 0, np.zeros(self.n_actions)
        q_values = self._habit_network.get_q_values(features[:84])
        action = int(np.argmax(q_values))
        return action, q_values

    def get_habit_q_values(self, sim=None) -> Optional[np.ndarray]:
        """
        Get Q-values from the habit network for current game state.

        Uses bridge (dual-process) if available, falls back to standalone DQN.

        Returns:
            6-dim array of action Q-values, or None if no habit loaded
        """
        # Prefer bridge (dual-process: SARSA + DQN)
        if self._bridge is not None and sim is not None:
            return self._bridge.get_action_values(sim)

        # Fallback to standalone habit network
        if self._habit_network is None or self._habit_encoder is None:
            return None
        if sim is not None:
            compressed = self._habit_encoder.encode_from_sim(sim)
            return self._habit_network.get_q_values(compressed)
        return None

    def load_bridge(
        self,
        sarsa_path: str = "brain/games/lolo/sarsa_qtable.npy",
        dqn_path: str = "brain/games/lolo/dqn_weights.pt",
    ) -> bool:
        """
        Load the dual-process SARSA↔DQN bridge.

        This supersedes the standalone habit network, providing:
          - Instant SARSA Q-lookup for known states
          - DQN generalization for unseen states
          - Continuous distillation (SARSA → DQN)
          - Live SARSA solving for new puzzles

        Args:
            sarsa_path: path to SARSA Q-table (.npy)
            dqn_path: path to DQN weights (.pt)

        Returns:
            True if at least one component loaded
        """
        try:
            from brain.games.lolo.sarsa_dqn_bridge import SarsaDQNBridge
            self._bridge = SarsaDQNBridge(n_actions=6)

            sarsa_ok = self._bridge.load_sarsa(sarsa_path)
            dqn_ok = self._bridge.load_dqn(dqn_path)

            if sarsa_ok or dqn_ok:
                self._habit_confidence = 1.0
                # Wire as dream policy too
                self._policy_fn = self._bridge_policy_fn
                return True
            else:
                self._bridge = None
                return False
        except Exception as e:
            print(f"  Bridge load failed: {e}")
            self._bridge = None
            return False

    def _bridge_policy_fn(self, features: np.ndarray, explore: bool = False):
        """Policy function using the bridge for dream action selection."""
        if self._bridge is None:
            return 0, np.zeros(self.n_actions)
        # Use bridge's DQN for dreaming (faster than SARSA lookup)
        if self._bridge._dqn is not None:
            q_values = self._bridge._dqn.get_q_values(features[:84])
            return int(np.argmax(q_values)), q_values
        return 0, np.zeros(self.n_actions)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context recognition and schedule dream simulations.

        Expected inputs:
            features: np.ndarray — state features (84-dim from CNN/FFT)
            reward: float — current reward
            step: int — current step count
            sim: optional LoloSimulator for habit network encoding
        """
        features = inputs.get("features")
        reward = inputs.get("reward", 0.0)
        sim = inputs.get("sim")  # Game-specific simulator for habit encoding

        # Context matching via SNN placeholder
        self._context_score = self._compute_context(features, reward)

        # ── Habit Network Q-values (instant, no dreaming needed) ──────
        habit_q = self.get_habit_q_values(sim=sim)

        # Check if we should dream
        self._steps_since_dream += 1
        should_dream = self._steps_since_dream >= self.dream_interval

        dream_results = None
        dream_q = None

        if should_dream and features is not None:
            self._steps_since_dream = 0

            # Run dream through world model
            dream_results = self._run_dream(features)
            dream_q = self._last_action_values

            if dream_results:
                self._total_dreams += 1

                # Send dreams to Amygdala for threat assessment
                self.send(
                    target="amygdala_thalamus",
                    msg_type="dream_results",
                    payload={"dream_results": dream_results},
                )
                # Send to Hippocampus for storage
                self.send(
                    target="hippocampus",
                    msg_type="dream_data",
                    payload={"dream_results": dream_results},
                )

        # ── Merge habit + dream action values ─────────────────────────
        action_values = self._merge_action_values(habit_q, dream_q)

        return {
            "context_score": self._context_score,
            "dream_results": dream_results,
            "wm_confidence": self._wm_confidence,
            "dream_action_values": action_values,
            "habit_active": habit_q is not None,
            "habit_confidence": self._habit_confidence,
        }

    def _merge_action_values(self, habit_q: Optional[np.ndarray],
                              dream_q: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Blend habit Q-values with dream Q-values.

        If only habit: return habit (padded to n_actions if needed)
        If only dream: return dream
        If both: weighted blend based on habit_weight
        """
        if habit_q is None and dream_q is None:
            return None

        if habit_q is not None and dream_q is None:
            # Pad habit Q (6 actions) to full action space if needed
            if len(habit_q) < self.n_actions:
                padded = np.zeros(self.n_actions, dtype=np.float32)
                padded[:len(habit_q)] = habit_q
                return padded
            return habit_q

        if habit_q is None and dream_q is not None:
            return dream_q

        # Both present: weighted blend
        w = self._habit_weight * self._habit_confidence
        # Pad habit if needed
        if len(habit_q) < self.n_actions:
            padded = np.zeros(self.n_actions, dtype=np.float32)
            padded[:len(habit_q)] = habit_q
            habit_q = padded

        # Normalize both to [0,1] range before blending
        h_range = habit_q.max() - habit_q.min()
        d_range = dream_q.max() - dream_q.min()
        h_norm = (habit_q - habit_q.min()) / max(h_range, 1e-8)
        d_norm = (dream_q - dream_q.min()) / max(d_range, 1e-8)

        return w * h_norm + (1.0 - w) * d_norm

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update WorldModel and habit network from real experience.

        Expected experience:
            state: np.ndarray — current features
            action: int
            next_state: np.ndarray — next features
            reward: float
            done: bool
            sim: optional LoloSimulator for habit fine-tuning
        """
        state = experience.get("state")
        action = experience.get("action")
        next_state = experience.get("next_state")
        reward = experience.get("reward", 0.0)
        done = experience.get("done", False)

        # ── Fine-tune habit network from real gameplay ─────────────────
        if self._habit_network is not None:
            sim = experience.get("sim")
            if sim is not None:
                self._habit_network.set_simulator(sim)
            # Feed the experience through the DQN's step (implicit learn)
            self._habit_network.step(
                obs=state, prev_action=action,
                reward=reward, done=done,
            )

        # ── World model training ──────────────────────────────────────
        if state is None or self._world_model is None:
            self._wm_train_steps += 1
            self._wm_confidence = min(1.0, self._wm_train_steps / 50.0)
            return {
                "wm_train_steps": self._wm_train_steps,
                "wm_confidence": self._wm_confidence,
            }

        # Store real transition in world model's replay
        self._world_model.store_transition(state, action, next_state, reward)

        # Throttled: skip gradient update if requested
        if experience.get("skip_train", False):
            self._wm_train_steps += 1
            return {"wm_loss": 0.0, "wm_train_steps": self._wm_train_steps}

        # Train world model
        result = self._world_model.train_step()

        self._wm_train_steps += 1
        self._wm_confidence = self._world_model.confidence

        return {
            **result,
            "wm_train_steps": self._wm_train_steps,
            "wm_confidence": round(self._wm_confidence, 3),
        }

    def _compute_context(self, features: Optional[np.ndarray], reward: float) -> float:
        """Simple context score placeholder. Full SNN integration later."""
        if features is None:
            return 0.0
        # Simple heuristic: normalized mean activation
        return float(np.clip(np.mean(np.abs(features)) * 0.5, 0, 1))

    def _run_dream(self, features: np.ndarray) -> list:
        """
        Run dream simulation through the world model.

        Uses the learned dynamics model to simulate future trajectories.
        Returns dream steps with predicted features, rewards, and values.
        """
        if self._world_model is None or not self._world_model.is_ready:
            return []

        # ── Dream a trajectory using policy ───────────────────────────
        trajectory = self._world_model.dream(
            start_features=features,
            horizon=self.dream_horizon,
            policy_fn=self._policy_fn,
        )

        # ── Evaluate all actions from current state ───────────────────
        # This produces action biases the Striatum can use
        self._last_action_values = self._world_model.dream_all_actions(
            features=features,
            depth=min(3, self.dream_horizon),
        )

        # Summary stats
        if trajectory:
            total_value = sum(t["predicted_reward"] for t in trajectory)
            self._last_dream_value = total_value

        return trajectory

    def report(self) -> Dict[str, Any]:
        base = super().report()
        wm_info = {}
        if self._world_model is not None:
            wm_info = self._world_model.report()
        habit_info = {}
        if self._habit_network is not None:
            habit_info = {
                "habit_" + k: v
                for k, v in self._habit_network.report().items()
            }
        return {
            **base,
            "context_score": round(self._context_score, 3),
            "wm_confidence": round(self._wm_confidence, 3),
            "wm_train_steps": self._wm_train_steps,
            "dream_interval": self.dream_interval,
            "dream_horizon": self.dream_horizon,
            "total_dreams": self._total_dreams,
            "last_dream_value": round(self._last_dream_value, 3),
            "has_world_model": self._world_model is not None,
            "has_policy_fn": self._policy_fn is not None,
            "has_habit_network": self._habit_network is not None,
            "habit_confidence": self._habit_confidence,
            **wm_info,
            **habit_info,
        }
