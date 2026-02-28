"""
basal_ganglia.py — Procedure Selection & Habit Learning Region.

Wraps:
  - MiniSNN (400-neuron spiking neural network for context matching)
  - WorldModel (learned dynamics model for dream simulation)
  - CompressedState encoder

Responsible for:
  - Context recognition via SNN (familiar vs novel states)
  - Running dream simulations via world model look-ahead
  - Procedure/habit selection based on learned patterns
  - Feeding dream results to Amygdala and Hippocampus
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

    def set_policy_fn(self, policy_fn: Callable) -> None:
        """
        Set the DQN policy function for dream action selection.

        Args:
            policy_fn: callable(features, explore=False) -> (action, q_values)
        """
        self._policy_fn = policy_fn

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context recognition and schedule dream simulations.

        Expected inputs:
            features: np.ndarray — state features (84-dim from CNN/FFT)
            reward: float — current reward
            step: int — current step count
        """
        features = inputs.get("features")
        reward = inputs.get("reward", 0.0)

        # Context matching via SNN placeholder
        self._context_score = self._compute_context(features, reward)

        # Check if we should dream
        self._steps_since_dream += 1
        should_dream = self._steps_since_dream >= self.dream_interval

        dream_results = None
        action_values = None

        if should_dream and features is not None:
            self._steps_since_dream = 0

            # Run dream through world model
            dream_results = self._run_dream(features)
            action_values = self._last_action_values

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

        return {
            "context_score": self._context_score,
            "dream_results": dream_results,
            "wm_confidence": self._wm_confidence,
            "dream_action_values": action_values,
        }

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update WorldModel from real experience.

        Expected experience:
            state: np.ndarray — current features
            action: int
            next_state: np.ndarray — next features
            reward: float
        """
        state = experience.get("state")
        action = experience.get("action")
        next_state = experience.get("next_state")
        reward = experience.get("reward", 0.0)

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
            **wm_info,
        }
