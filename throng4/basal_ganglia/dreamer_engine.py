"""
DreamerEngine — Basal Ganglia dream simulation engine.

Runs multiple lightweight world models in parallel, each testing a different
hypothesis/policy. Uses throng2's SNN architecture for fast forward simulation.

The dreamer does NOT make decisions — it produces ranked evaluations.
The Amygdala and PolicyMonitor act on these evaluations.

Network size tiers:
  - micro (50 neurons):  real-time per-step dreaming, ~0.5ms/step
  - mini  (200 neurons): between-episode planning, ~2ms/step
  - full  (1000+ neurons): deep offline analysis, ~20ms/step
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple
from enum import Enum

from throng4.basal_ganglia.compressed_state import (
    CompressedStateEncoder,
    CompressedState,
    EncodingMode,
)


class NetworkSize(Enum):
    """Network size tier for dreamer simulations."""
    MICRO = "micro"    # 50 neurons, fastest
    MINI = "mini"      # 200 neurons, balanced
    FULL = "full"      # 1000+ neurons, most accurate


# Network configurations per tier
NETWORK_CONFIGS = {
    NetworkSize.MICRO: {
        'n_neurons': 50,
        'hidden_sizes': [16],
        'connection_prob': 0.1,
    },
    NetworkSize.MINI: {
        'n_neurons': 200,
        'hidden_sizes': [32],
        'connection_prob': 0.05,
    },
    NetworkSize.FULL: {
        'n_neurons': 1000,
        'hidden_sizes': [64, 32],
        'connection_prob': 0.02,
    },
}


@dataclass
class DreamResult:
    """Result from a single hypothesis dream simulation."""
    hypothesis_id: int
    hypothesis_name: str
    predicted_rewards: List[float]       # Reward at each simulated step
    total_predicted_reward: float        # Sum of predicted rewards
    avg_predicted_reward: float          # Mean reward per step
    worst_step_reward: float             # Minimum reward in sequence
    best_step_reward: float              # Maximum reward in sequence
    confidence: float                    # 0-1, how confident the sim is
    simulation_time_ms: float            # How long the dream took
    final_state: Optional[np.ndarray] = None  # Compressed final state
    trajectory: List[int] = field(default_factory=list)  # Actions taken

    @property
    def is_positive(self) -> bool:
        """Does this hypothesis predict overall positive outcomes?"""
        return self.total_predicted_reward > 0

    @property
    def has_catastrophe(self) -> bool:
        """Does any step show severe negative reward?"""
        return self.worst_step_reward < -1.0

    def summary(self) -> str:
        emoji = "✅" if self.is_positive else "⚠️"
        return (
            f"{emoji} H{self.hypothesis_id} ({self.hypothesis_name}): "
            f"total={self.total_predicted_reward:+.2f}, "
            f"worst={self.worst_step_reward:+.2f}, "
            f"conf={self.confidence:.2f}, "
            f"time={self.simulation_time_ms:.1f}ms"
        )


@dataclass
class Hypothesis:
    """A policy hypothesis to simulate."""
    id: int
    name: str
    action_selector: Callable[[np.ndarray], int]  # state → action
    description: str = ""


class WorldModel:
    """
    Lightweight learned world model for dream simulation.

    Uses a simple feedforward network (inspired by throng2's LayeredNetwork)
    to predict next_state and reward from (state, action).

    This is intentionally simpler than the full SNN — speed over fidelity.
    """

    def __init__(self, state_size: int, n_actions: int,
                 config: dict = None):
        config = config or NETWORK_CONFIGS[NetworkSize.MICRO]
        hidden = config['hidden_sizes'][0]

        # State transition model: (state + action_onehot) → next_state
        input_size = state_size + n_actions
        self.w1 = np.random.randn(input_size, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.randn(hidden, state_size) * 0.1
        self.b2 = np.zeros(state_size)

        # Reward prediction: (state + action_onehot) → reward
        self.wr1 = np.random.randn(input_size, hidden) * 0.1
        self.br1 = np.zeros(hidden)
        self.wr2 = np.random.randn(hidden, 1) * 0.1
        self.br2 = np.zeros(1)

        self.state_size = state_size
        self.n_actions = n_actions
        self._train_count = 0

    def predict(self, state: np.ndarray, action: int
                ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward.

        Args:
            state: Current compressed state
            action: Action index

        Returns:
            (predicted_next_state, predicted_reward)
        """
        # Build input: concat state + one-hot action
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action] = 1.0
        x = np.concatenate([state.flatten(), action_onehot])

        # State prediction
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        next_state = h @ self.w2 + self.b2

        # Reward prediction
        hr = np.maximum(0, x @ self.wr1 + self.br1)
        reward = float((hr @ self.wr2 + self.br2)[0])

        return next_state, reward

    def update(self, state: np.ndarray, action: int,
               actual_next_state: np.ndarray, actual_reward: float,
               lr: float = 0.001):
        """
        Update world model from real experience.

        Simple gradient descent on prediction error.
        """
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action] = 1.0
        x = np.concatenate([state.flatten(), action_onehot])

        # Forward pass — state
        h = np.maximum(0, x @ self.w1 + self.b1)
        pred_state = h @ self.w2 + self.b2

        # State error
        state_error = pred_state - actual_next_state.flatten()
        if state_error.size != self.state_size:
            # Dimension mismatch — skip this update
            return

        # Backward pass — state prediction
        dw2 = np.outer(h, state_error)
        db2 = state_error
        dh = state_error @ self.w2.T
        dh = dh * (h > 0)  # ReLU derivative
        dw1 = np.outer(x, dh)
        db1 = dh

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        # Forward pass — reward
        hr = np.maximum(0, x @ self.wr1 + self.br1)
        pred_reward = float((hr @ self.wr2 + self.br2)[0])

        # Reward error
        reward_error = pred_reward - actual_reward
        dwr2 = hr.reshape(-1, 1) * reward_error
        dbr2 = np.array([reward_error])
        dhr = reward_error * self.wr2.flatten()
        dhr = dhr * (hr > 0)
        dwr1 = np.outer(x, dhr)
        dbr1 = dhr

        self.wr2 -= lr * dwr2
        self.br2 -= lr * dbr2
        self.wr1 -= lr * dwr1
        self.br1 -= lr * dbr1

        self._train_count += 1

    @property
    def is_calibrated(self) -> bool:
        """Has the model seen enough data to be useful?
        
        Threshold lowered from 50 to 10: early Tetris episodes end in 2-5
        pieces, so 50 steps takes many episodes to accumulate. The world
        model is imperfect at 10 steps but still useful for hypothesis ranking.
        """
        return self._train_count >= 10

    def get_confidence(self) -> float:
        """Confidence based on training experience.
        
        Denominator lowered from 500 to 50: Tetris episodes are 2-5 steps,
        so 500 steps takes 100+ episodes to reach 0.9. At 50, we reach
        0.9 confidence after ~45 steps (~10 episodes), making the dreamer
        useful much sooner in training.
        """
        return min(0.9, self._train_count / 50.0)


class DreamerEngine:
    """
    Basal Ganglia dream simulation engine.

    Runs multiple hypotheses through a learned world model to predict
    future outcomes. Does NOT make decisions — only produces evaluations.

    Usage:
        dreamer = DreamerEngine(n_hypotheses=3, network_size='micro')
        dreamer.learn(state, action, next_state, reward)  # update world model
        results = dreamer.dream(current_state, hypotheses, n_steps=10)
        # results is List[DreamResult], ranked by predicted reward
    """

    def __init__(self,
                 n_hypotheses: int = 3,
                 network_size: str = 'micro',
                 state_size: int = 64,
                 n_actions: int = 4,
                 dream_interval: int = 5):
        """
        Args:
            n_hypotheses: Max parallel hypotheses to simulate
            network_size: 'micro', 'mini', or 'full'
            state_size: Compressed state dimension
            n_actions: Number of possible actions
            dream_interval: Dream every N steps (0 = every step)
        """
        self.n_hypotheses = n_hypotheses
        self.network_size = NetworkSize(network_size)
        self.config = NETWORK_CONFIGS[self.network_size]
        self.state_size = state_size
        self.n_actions = n_actions
        self.dream_interval = dream_interval

        # World model (shared across hypotheses)
        self.world_model = WorldModel(state_size, n_actions, self.config)

        # State encoder
        self.encoder = CompressedStateEncoder(
            mode=EncodingMode.QUANTIZED,
            n_quantize_levels=4,
        )

        # Stats
        self._step_count = 0
        self._dream_count = 0
        self._total_dream_time_ms = 0.0
        self._last_dream_results: List[DreamResult] = []

    def learn(self, state: np.ndarray, action: int,
              next_state: np.ndarray, reward: float):
        """
        Update the world model from real experience.

        Call this on every real environment step.
        """
        # Compress states for the world model
        cs = self._ensure_compressed(state)
        cs_next = self._ensure_compressed(next_state)

        self.world_model.update(cs, action, cs_next, reward)
        self._step_count += 1

    def dream(self, current_state: np.ndarray,
              hypotheses: List[Hypothesis],
              n_steps: int = 10) -> List[DreamResult]:
        """
        Simulate multiple hypotheses forward from current state.

        Args:
            current_state: Current observation (will be compressed)
            hypotheses: List of policy hypotheses to test
            n_steps: How many steps to simulate ahead

        Returns:
            List[DreamResult] sorted by total_predicted_reward (best first)
        """
        compressed = self._ensure_compressed(current_state)
        results = []

        for hyp in hypotheses[:self.n_hypotheses]:
            t0 = time.perf_counter()
            result = self._simulate_hypothesis(compressed, hyp, n_steps)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result.simulation_time_ms = elapsed_ms
            results.append(result)

        # Sort by predicted total reward (best first)
        results.sort(key=lambda r: r.total_predicted_reward, reverse=True)

        self._dream_count += 1
        self._total_dream_time_ms += sum(r.simulation_time_ms for r in results)
        self._last_dream_results = results

        return results

    def should_dream(self) -> bool:
        """Check if it's time for a dream cycle based on interval."""
        if self.dream_interval <= 0:
            return True
        return self._step_count % self.dream_interval == 0

    def _simulate_hypothesis(self, initial_state: np.ndarray,
                             hypothesis: Hypothesis,
                             n_steps: int) -> DreamResult:
        """Run a single hypothesis through the world model."""
        state = initial_state.copy()
        rewards = []
        actions = []

        for step in range(n_steps):
            # Hypothesis selects action
            try:
                action = hypothesis.action_selector(state)
                action = int(action) % self.n_actions
            except Exception:
                action = np.random.randint(self.n_actions)

            # World model predicts outcome
            next_state, predicted_reward = self.world_model.predict(
                state, action
            )

            rewards.append(predicted_reward)
            actions.append(action)
            state = next_state

        total = sum(rewards)
        avg = total / max(len(rewards), 1)

        return DreamResult(
            hypothesis_id=hypothesis.id,
            hypothesis_name=hypothesis.name,
            predicted_rewards=rewards,
            total_predicted_reward=total,
            avg_predicted_reward=avg,
            worst_step_reward=min(rewards) if rewards else 0.0,
            best_step_reward=max(rewards) if rewards else 0.0,
            confidence=self.world_model.get_confidence(),
            simulation_time_ms=0.0,  # Set by caller
            final_state=state,
            trajectory=actions,
        )

    def _ensure_compressed(self, obs: np.ndarray) -> np.ndarray:
        """Ensure observation is compressed to state_size."""
        flat = obs.flatten().astype(np.float32)
        if flat.size == self.state_size:
            return flat
        elif flat.size > self.state_size:
            # Downsample
            indices = np.linspace(0, flat.size - 1,
                                  self.state_size, dtype=int)
            return flat[indices]
        else:
            # Pad with zeros
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:flat.size] = flat
            return padded

    def create_default_hypotheses(self, n_actions: int) -> List[Hypothesis]:
        """
        Create basic hypothesis set for testing.

        Returns 3 hypotheses:
          - Greedy: always pick action 0 (exploitation baseline)
          - Random: random actions (exploration baseline)
          - Biased: slight preference for certain actions
        """
        hypotheses = [
            Hypothesis(
                id=0,
                name="greedy",
                action_selector=lambda s: 0,
                description="Always pick first action (exploitation baseline)",
            ),
            Hypothesis(
                id=1,
                name="random",
                action_selector=lambda s: np.random.randint(n_actions),
                description="Random actions (exploration baseline)",
            ),
            Hypothesis(
                id=2,
                name="biased",
                action_selector=lambda s: np.argmax(s[:n_actions])
                if s.size >= n_actions
                else np.random.randint(n_actions),
                description="Action biased by state features",
            ),
        ]
        return hypotheses

    @property
    def last_dream_results(self) -> List[DreamResult]:
        return self._last_dream_results

    @property
    def is_calibrated(self) -> bool:
        return self.world_model.is_calibrated

    @property
    def avg_dream_time_ms(self) -> float:
        if self._dream_count == 0:
            return 0.0
        return self._total_dream_time_ms / self._dream_count

    def summary(self) -> str:
        lines = [
            f"DreamerEngine ({self.network_size.value}):",
            f"  Steps observed: {self._step_count}",
            f"  Dreams run: {self._dream_count}",
            f"  Avg dream time: {self.avg_dream_time_ms:.1f}ms",
            f"  World model calibrated: {self.is_calibrated}",
            f"  World model confidence: "
            f"{self.world_model.get_confidence():.2f}",
        ]
        if self._last_dream_results:
            lines.append("  Last dream results:")
            for r in self._last_dream_results:
                lines.append(f"    {r.summary()}")
        return "\n".join(lines)
