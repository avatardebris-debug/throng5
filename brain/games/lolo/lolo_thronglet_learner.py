"""
lolo_thronglet_learner.py — Throng2 neural network learner for Lolo.

Uses the Thronglet architecture (Fibonacci geometry, small-world
connectivity, Hebbian learning, Nash equilibrium pruning) from Throng2
for fast training on the Lolo simulator.

The Throng2 NN achieved 65% on Morris water maze with 2000 neurons.
For Lolo, we use a smaller network (1000-2000 neurons) since the
observation space is the compact grid state.

Same step() API as WholeBrain — drop-in replacement for LoloCurriculum.

Transfer artifacts:
  - Trained inter-layer weights (→ DQN initialization)
  - Discovered state representations (→ compressed state encoder)
  - Action sequences (→ skill library proven chains)
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import Throng2 from throng4_new
_T2_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "throng4_new", "throng2-master",
)
if os.path.isdir(_T2_ROOT):
    sys.path.insert(0, _T2_ROOT)

try:
    from src.core.network import LayeredNetwork, ThrongletNetwork
    from src.learning.nash_pruning import NashPruningSystem
    from src.learning.neuromodulators import NeuromodulatorSystem
    THRONG2_AVAILABLE = True
except ImportError:
    THRONG2_AVAILABLE = False


class LoloThrongletLearner:
    """
    Throng2-based learner for Lolo puzzles.

    Architecture:
      Input (217 raw obs) → Hidden [500, 250] → Output (6 actions)
      Total: ~970 neurons with Fibonacci geometry + small-world connections

    Learning:
      - Dopamine-modulated Hebbian (TD error → learning rate)
      - Nash equilibrium pruning (every 100 episodes)
      - Neuromodulator system (dopamine, serotonin, norepinephrine)

    Provides the same step() API as WholeBrain.
    """

    def __init__(
        self,
        n_actions: int = 6,
        n_hidden: int = 1000,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        prune_interval: int = 100,
        learn_interval: int = 10,  # Batch Hebbian: learn every N steps
    ):
        if not THRONG2_AVAILABLE:
            raise ImportError(
                "Throng2 not found. Expected at: " + _T2_ROOT
            )

        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._prune_interval = prune_interval
        self._learn_interval = learn_interval

        # ── Thronglet network ─────────────────────────────────────────
        # Input: 217-dim raw obs, Output: 6 actions
        # Hidden: split into 2 layers for hierarchical processing
        h1 = n_hidden // 2
        h2 = n_hidden // 4
        self.network = LayeredNetwork(
            input_size=217,
            hidden_sizes=[h1, h2],
            output_size=n_actions,
            dimension=2,
            connection_prob=0.03,  # Slightly denser for smaller network
        )

        # ── Neuromodulators ───────────────────────────────────────────
        self.neuromod = NeuromodulatorSystem()

        # ── Nash pruning ──────────────────────────────────────────────
        self.pruner = NashPruningSystem(
            pruning_threshold=0.08,
            resource_budget=1.0,
            competition_strength=0.5,
        )

        # ── Q-value readout layer (simple linear for action selection) ─
        # The thronglet network output spikes are noisy — smooth with
        # a learned linear readout
        self._readout_w = np.random.randn(n_actions, n_actions) * 0.1
        self._readout_lr = 0.01

        # ── State ─────────────────────────────────────────────────────
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_actions: List[int] = []
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_action: int = 0
        self._prev_spikes: Optional[np.ndarray] = None

        # ── Transfer artifacts ────────────────────────────────────────
        self.success_chains: List[List[int]] = []
        self.visited_states: set = set()

        # ── Stats ─────────────────────────────────────────────────────
        self._recent_rewards: List[float] = []
        self._pruning_stats: List[Dict] = []

        # Placeholder for duck-typing as fast learner
        self.q_table = {}  # Enables curriculum fast-learner detection

    def step(
        self,
        obs: Any,
        prev_action: int = 0,
        reward: float = 0.0,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Same API as WholeBrain.step()."""
        obs = np.asarray(obs, dtype=np.float32)

        # Clamp inputs to reasonable range
        obs_clamped = np.clip(obs, -3.0, 3.0)

        # Pad/truncate to 217
        if len(obs_clamped) < 217:
            obs_clamped = np.pad(obs_clamped, (0, 217 - len(obs_clamped)))
        elif len(obs_clamped) > 217:
            obs_clamped = obs_clamped[:217]

        self._total_steps += 1
        self._episode_reward += reward

        # ── Hash state for tracking ───────────────────────────────────
        state_hash = hash(np.round(obs[:20] * 3).astype(np.int8).tobytes())
        self.visited_states.add(state_hash)

        # ── Learning (batch: every N steps or on done) ────────────────
        if self._prev_obs is not None:
            # Compute TD error as dopamine signal
            td_error = self.neuromod.compute_td_error(
                tuple(self._prev_obs[:4].tolist()),  # State tuple
                reward,
                tuple(obs_clamped[:4].tolist()),      # Next state
                done,
            )

            # Hebbian learning — batched for speed (every N steps or on done)
            if done or (self._total_steps % self._learn_interval == 0):
                lr = self.neuromod.modulate_hebbian(base_rate=0.01)
                self.network.learn(
                    learning_rate=lr,
                    modulation=self.neuromod.dopamine,
                )

            # Update readout weights (simple gradient step)
            if self._prev_spikes is not None:
                spikes = self._prev_spikes[-self.n_actions:]
                target = np.zeros(self.n_actions)
                target[self._prev_action] = reward + 0.99 * np.max(
                    self._readout_w @ obs_clamped[-self.n_actions:]
                    if len(obs_clamped) >= self.n_actions
                    else np.zeros(self.n_actions)
                )
                q_pred = self._readout_w @ spikes
                error = target - q_pred
                self._readout_w += self._readout_lr * np.outer(
                    error, spikes
                )
                self._readout_w = np.clip(self._readout_w, -2.0, 2.0)

        # ── Forward pass ──────────────────────────────────────────────
        output_spikes = self.network.forward(obs_clamped)
        self._prev_spikes = output_spikes.copy()

        # ── Action selection ──────────────────────────────────────────
        action_spikes = output_spikes[-self.n_actions:]

        if np.random.random() < self.epsilon:
            # Explore — but bias toward spike-suggested direction
            if np.sum(action_spikes) > 0.5 and np.random.random() < 0.3:
                probs = action_spikes / np.sum(action_spikes)
                action = int(np.random.choice(self.n_actions, p=probs))
            else:
                action = int(np.random.randint(self.n_actions))
        else:
            # Exploit — use readout layer
            q_vals = self._readout_w @ action_spikes
            if np.max(q_vals) > np.min(q_vals) + 0.01:
                action = int(np.argmax(q_vals))
            elif np.sum(action_spikes) > 0:
                probs = action_spikes / np.sum(action_spikes)
                action = int(np.random.choice(self.n_actions, p=probs))
            else:
                action = int(np.random.randint(self.n_actions))

        self._episode_actions.append(action)

        # ── Episode boundary ──────────────────────────────────────────
        if done:
            self._total_episodes += 1
            self.epsilon = max(self.epsilon_end,
                              self.epsilon * self.epsilon_decay)
            self._recent_rewards.append(self._episode_reward)
            if len(self._recent_rewards) > 100:
                self._recent_rewards = self._recent_rewards[-100:]

            # Save successful chains
            if self._episode_reward > 0:
                if len(self._episode_actions) < 500:
                    self.success_chains.append(self._episode_actions.copy())
                    if len(self.success_chains) > 100:
                        self.success_chains = self.success_chains[-100:]

            # Nash pruning periodically
            if (self._total_episodes % self._prune_interval == 0
                    and self._total_episodes > 0):
                self._run_nash_pruning()

            # Reset episode state
            self._episode_actions = []
            self._episode_reward = 0.0
            self._prev_obs = None
            self._prev_action = 0
            self.network.reset()
        else:
            self._prev_obs = obs_clamped
            self._prev_action = action

        return {"action": action}

    def _run_nash_pruning(self):
        """Run Nash equilibrium pruning on the hidden layers."""
        for layer in self.network.layers[1:-1]:  # Hidden layers only
            activities = layer.neurons.get_activity_rates()
            rewards_arr = np.full(layer.n_neurons,
                                 np.mean(self._recent_rewards)
                                 if self._recent_rewards else 0.0)

            if not layer.use_sparse:
                pruned_w, stats = self.pruner.prune_network(
                    layer.weights, activities, rewards_arr,
                )
                layer.weights = pruned_w
                self._pruning_stats.append(stats)

                # Allow regrowth
                layer.weights = self.pruner.allow_regrowth(
                    layer.weights, activities, growth_rate=0.005,
                )

    def report(self) -> Dict[str, Any]:
        """Report learner stats."""
        layer_info = []
        for i, layer in enumerate(self.network.layers):
            n = layer.n_neurons
            if layer.use_sparse:
                conns = layer.weights.nnz
            else:
                conns = int(np.count_nonzero(layer.weights))
            layer_info.append({"layer": i, "neurons": n, "connections": conns})

        return {
            "type": "LoloThrongletLearner",
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "epsilon": round(self.epsilon, 4),
            "visited_states": len(self.visited_states),
            "success_chains": len(self.success_chains),
            "layers": layer_info,
            "avg_reward_100": round(
                np.mean(self._recent_rewards[-100:]) if self._recent_rewards else 0, 2
            ),
            "pruning_rounds": len(self._pruning_stats),
        }

    def close(self):
        pass

    def get_transfer_data(self) -> Dict[str, Any]:
        """Extract trained weights and knowledge for WholeBrain transfer."""
        return {
            "layer_weights": [w.copy() for w in self.network.layer_weights],
            "readout_weights": self._readout_w.copy(),
            "chains": self.success_chains[-50:],
            "landmarks": list(self.visited_states)[:1000],
            "stats": self.report(),
        }
