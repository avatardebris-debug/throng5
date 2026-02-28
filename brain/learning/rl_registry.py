"""
rl_registry.py — Registry of available RL learners for the Striatum.

Catalogs algorithms from deep_rl_zoo and our built-in numpy DQN,
each with metadata about their capabilities and resource requirements.

Usage:
    from brain.learning.rl_registry import RLRegistry

    registry = RLRegistry()
    registry.list_all()
    info = registry.get("rainbow")
    learner_cls = registry.create("dqn_builtin", n_features=84, n_actions=18)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np


# ── Algorithm Classification ─────────────────────────────────────────

class AlgorithmType(Enum):
    VALUE_BASED = "value_based"           # DQN, Rainbow, C51, IQN
    POLICY_GRADIENT = "policy_gradient"    # REINFORCE, A2C
    ACTOR_CRITIC = "actor_critic"          # PPO, SAC, IMPALA
    HYBRID = "hybrid"                     # Agent57, NGU


class ActionSpaceType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    BOTH = "both"


# ── Learner Interface ─────────────────────────────────────────────────

class Learner(ABC):
    """
    Common interface for all RL learners used by the Striatum.

    Every learner (built-in or RLZoo-wrapped) implements this interface
    so the Striatum can swap algorithms at runtime.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select an action given current state and exploration rate."""
        ...

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Update from a single transition. Returns metrics dict."""
        ...

    @abstractmethod
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values (or action preferences) for all actions."""
        ...

    @abstractmethod
    def save_weights(self) -> Dict[str, Any]:
        """Serialize learner weights for checkpointing."""
        ...

    @abstractmethod
    def load_weights(self, weights: Dict[str, Any]) -> None:
        """Load learner weights from checkpoint."""
        ...

    def statistics(self) -> Dict[str, Any]:
        """Return current learner statistics."""
        return {}


# ── Algorithm Metadata ────────────────────────────────────────────────

@dataclass
class AlgorithmInfo:
    """Metadata about a registered RL algorithm."""
    name: str
    display_name: str
    algorithm_type: AlgorithmType
    action_space: ActionSpaceType

    # Performance characteristics (1-5 scale)
    sample_efficiency: int = 3       # Higher = needs fewer environment steps
    compute_cost: int = 3            # Higher = more GPU/CPU time per step
    stability: int = 3               # Higher = more stable training

    # Feature flags
    requires_torch: bool = False     # Needs PyTorch
    supports_prioritized_replay: bool = False
    supports_n_step: bool = False
    supports_distributional: bool = False
    supports_recurrent: bool = False

    # Source
    source: str = "builtin"          # "builtin", "deep_rl_zoo", "custom"
    source_path: str = ""            # Path to source module

    # Factory (optional — set when algorithm is registered)
    factory: Optional[Callable] = field(default=None, repr=False)

    # Notes for the selector
    best_for: str = ""               # Human-readable note on best use case


# ── Registry ──────────────────────────────────────────────────────────

class RLRegistry:
    """
    Registry of available RL algorithms.

    Populated with both built-in numpy implementations and
    deep_rl_zoo PyTorch algorithms.
    """

    def __init__(self):
        self._algorithms: Dict[str, AlgorithmInfo] = {}
        self._register_defaults()

    def register(self, info: AlgorithmInfo) -> None:
        """Register an algorithm."""
        self._algorithms[info.name] = info

    def get(self, name: str) -> Optional[AlgorithmInfo]:
        """Get algorithm info by name."""
        return self._algorithms.get(name)

    def list_all(self) -> List[AlgorithmInfo]:
        """List all registered algorithms."""
        return list(self._algorithms.values())

    def list_by_type(self, algo_type: AlgorithmType) -> List[AlgorithmInfo]:
        """List algorithms of a specific type."""
        return [a for a in self._algorithms.values() if a.algorithm_type == algo_type]

    def list_for_action_space(self, space: ActionSpaceType) -> List[AlgorithmInfo]:
        """List algorithms compatible with an action space type."""
        return [
            a for a in self._algorithms.values()
            if a.action_space == space or a.action_space == ActionSpaceType.BOTH
        ]

    def create(self, name: str, **kwargs) -> Optional[Learner]:
        """Create a learner instance from registry."""
        info = self._algorithms.get(name)
        if info is None or info.factory is None:
            return None
        return info.factory(**kwargs)

    def summary(self) -> str:
        """Human-readable summary of all registered algorithms."""
        lines = ["RL Algorithm Registry", "=" * 50]
        for info in self._algorithms.values():
            torch_tag = " [torch]" if info.requires_torch else " [numpy]"
            lines.append(
                f"  {info.name:20s} | {info.algorithm_type.value:16s} | "
                f"{info.action_space.value:10s} | eff={info.sample_efficiency} "
                f"cost={info.compute_cost}{torch_tag}"
            )
        return "\n".join(lines)

    # ── Default Registration ──────────────────────────────────────────

    def _register_defaults(self):
        """Register all known algorithms."""

        # ── Built-in (numpy, no dependencies) ─────────────────────────

        self.register(AlgorithmInfo(
            name="dqn_builtin",
            display_name="DQN (Built-in NumPy)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=2, compute_cost=1, stability=4,
            requires_torch=False,
            source="builtin",
            factory=_create_builtin_dqn,
            best_for="Fast prototyping, CPU-only, small state spaces",
        ))

        self.register(AlgorithmInfo(
            name="dqn_torch",
            display_name="DQN (Built-in PyTorch Dueling)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=2, stability=4,
            requires_torch=True,
            source="builtin",
            factory=_create_torch_dqn,
            best_for="Built-in Dueling DQN with Double DQN, requires PyTorch",
        ))

        # ── deep_rl_zoo algorithms (PyTorch) ──────────────────────────

        self.register(AlgorithmInfo(
            name="dqn",
            display_name="DQN (deep_rl_zoo)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=2, compute_cost=2, stability=4,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.dqn.agent",
            best_for="Baseline value-based learning",
        ))

        self.register(AlgorithmInfo(
            name="double_dqn",
            display_name="Double DQN",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=2, stability=4,
            requires_torch=True,
            supports_prioritized_replay=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.double_dqn.agent",
            best_for="Reduced overestimation, more stable than DQN",
        ))

        self.register(AlgorithmInfo(
            name="rainbow",
            display_name="Rainbow DQN",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=5, compute_cost=4, stability=4,
            requires_torch=True,
            supports_prioritized_replay=True,
            supports_n_step=True,
            supports_distributional=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.rainbow.agent",
            best_for="Best sample efficiency for discrete, compute-heavy",
        ))

        self.register(AlgorithmInfo(
            name="c51",
            display_name="C51 (Categorical DQN)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=4, compute_cost=3, stability=4,
            requires_torch=True,
            supports_distributional=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.c51_dqn.agent",
            best_for="Distributional RL, risk-sensitive policies",
        ))

        self.register(AlgorithmInfo(
            name="iqn",
            display_name="IQN (Implicit Quantile Network)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=4, compute_cost=3, stability=3,
            requires_torch=True,
            supports_distributional=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.iqn.agent",
            best_for="Flexible risk levels, distributional RL",
        ))

        self.register(AlgorithmInfo(
            name="qr_dqn",
            display_name="QR-DQN (Quantile Regression)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=4, compute_cost=3, stability=4,
            requires_torch=True,
            supports_distributional=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.qr_dqn.agent",
            best_for="Distributional RL with quantile regression",
        ))

        self.register(AlgorithmInfo(
            name="drqn",
            display_name="DRQN (Recurrent DQN)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=3, stability=3,
            requires_torch=True,
            supports_recurrent=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.drqn.agent",
            best_for="Partially observable environments (POMDPs)",
        ))

        self.register(AlgorithmInfo(
            name="r2d2",
            display_name="R2D2 (Recurrent Replay)",
            algorithm_type=AlgorithmType.VALUE_BASED,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=4, compute_cost=4, stability=3,
            requires_torch=True,
            supports_recurrent=True, supports_prioritized_replay=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.r2d2.agent",
            best_for="POMDPs with prioritized replay",
        ))

        self.register(AlgorithmInfo(
            name="ppo",
            display_name="PPO (Proximal Policy Optimization)",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.BOTH,
            sample_efficiency=2, compute_cost=3, stability=5,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.ppo.agent",
            best_for="Most stable policy gradient, continuous or discrete",
        ))

        self.register(AlgorithmInfo(
            name="ppo_icm",
            display_name="PPO + ICM (Intrinsic Curiosity)",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=4, stability=4,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.ppo_icm.agent",
            best_for="Sparse reward / exploration-heavy (Montezuma, Pitfall)",
        ))

        self.register(AlgorithmInfo(
            name="ppo_rnd",
            display_name="PPO + RND (Random Network Distillation)",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=4, stability=4,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.ppo_rnd.agent",
            best_for="Exploration via novelty prediction (Montezuma specialist)",
        ))

        self.register(AlgorithmInfo(
            name="a2c",
            display_name="A2C (Advantage Actor-Critic)",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=2, compute_cost=2, stability=3,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.a2c.agent",
            best_for="Simple actor-critic, lightweight",
        ))

        self.register(AlgorithmInfo(
            name="sac",
            display_name="SAC (Soft Actor-Critic)",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.CONTINUOUS,
            sample_efficiency=4, compute_cost=3, stability=5,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.sac.agent",
            best_for="Continuous control (MuJoCo), entropy-regularized",
        ))

        self.register(AlgorithmInfo(
            name="impala",
            display_name="IMPALA",
            algorithm_type=AlgorithmType.ACTOR_CRITIC,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=3, compute_cost=5, stability=3,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.impala.agent",
            best_for="Distributed training, high throughput",
        ))

        self.register(AlgorithmInfo(
            name="ngu",
            display_name="NGU (Never Give Up)",
            algorithm_type=AlgorithmType.HYBRID,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=5, compute_cost=5, stability=3,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.ngu.agent",
            best_for="Hard exploration (Montezuma), episodic curiosity",
        ))

        self.register(AlgorithmInfo(
            name="agent57",
            display_name="Agent57",
            algorithm_type=AlgorithmType.HYBRID,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=5, compute_cost=5, stability=3,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.agent57.agent",
            best_for="SOTA Atari exploration, very compute-heavy",
        ))

        self.register(AlgorithmInfo(
            name="reinforce",
            display_name="REINFORCE",
            algorithm_type=AlgorithmType.POLICY_GRADIENT,
            action_space=ActionSpaceType.DISCRETE,
            sample_efficiency=1, compute_cost=1, stability=2,
            requires_torch=True,
            source="deep_rl_zoo", source_path="deep_rl_zoo.reinforce.agent",
            best_for="Simplest policy gradient, educational/baseline",
        ))


# ── Built-in DQN Factory ─────────────────────────────────────────────

def _create_builtin_dqn(**kwargs) -> Learner:
    """Create the built-in numpy DQN learner."""
    return BuiltinDQN(**kwargs)


class BuiltinDQN(Learner):
    """
    Lightweight DQN using pure numpy (no PyTorch dependency).

    This is the default learner for the Striatum — fast, simple,
    works on CPU. For more sophisticated learning, swap to a
    RLZoo algorithm.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        hidden_size: int = 128,
        gamma: float = 0.99,
        lr: float = 0.001,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / hidden_size)

        self._W1 = rng.randn(n_features, hidden_size).astype(np.float32) * scale1
        self._b1 = np.zeros(hidden_size, dtype=np.float32)
        self._W2 = rng.randn(hidden_size, n_actions).astype(np.float32) * scale2
        self._b2 = np.zeros(n_actions, dtype=np.float32)

        self._tW1 = self._W1.copy()
        self._tb1 = self._b1.copy()
        self._tW2 = self._W2.copy()
        self._tb2 = self._b2.copy()

        from collections import deque
        self._replay: deque = deque(maxlen=buffer_size)
        self._total_updates = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        q = self.get_q_values(state)
        return int(np.argmax(q))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32).flatten()
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[:self.n_features]
        hidden = np.maximum(0, x @ self._W1 + self._b1)
        return hidden @ self._W2 + self._b2

    def update(self, state, action, reward, next_state, done):
        self._replay.append((
            np.asarray(state, dtype=np.float32),
            action, reward,
            np.asarray(next_state, dtype=np.float32),
            done,
        ))
        if len(self._replay) < self.batch_size:
            return {"loss": 0.0}

        indices = np.random.choice(len(self._replay), self.batch_size, replace=False)
        batch = [self._replay[i] for i in indices]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Online Q
        h = np.maximum(0, states @ self._W1 + self._b1)
        q = h @ self._W2 + self._b2
        q_sel = q[np.arange(self.batch_size), actions]

        # Target Q
        th = np.maximum(0, next_states @ self._tW1 + self._tb1)
        tq = th @ self._tW2 + self._tb2
        q_target = rewards + self.gamma * np.max(tq, axis=1) * (1 - dones)

        td = q_target - q_sel
        loss = float(np.mean(td ** 2))

        # Backward
        dQ = np.zeros_like(q)
        dQ[np.arange(self.batch_size), actions] = -2 * td / self.batch_size
        dW2 = h.T @ dQ
        db2 = np.sum(dQ, axis=0)
        dh = dQ @ self._W2.T
        dh[h <= 0] = 0
        dW1 = states.T @ dh
        db1 = np.sum(dh, axis=0)

        self._W1 -= self.lr * dW1
        self._b1 -= self.lr * db1
        self._W2 -= self.lr * dW2
        self._b2 -= self.lr * db2

        self._total_updates += 1
        if self._total_updates % self.target_update_freq == 0:
            self._tW1, self._tb1 = self._W1.copy(), self._b1.copy()
            self._tW2, self._tb2 = self._W2.copy(), self._b2.copy()

        return {"loss": loss, "td_error": float(np.mean(np.abs(td)))}

    def save_weights(self):
        return {"W1": self._W1, "b1": self._b1, "W2": self._W2, "b2": self._b2}

    def load_weights(self, weights):
        self._W1, self._b1 = weights["W1"], weights["b1"]
        self._W2, self._b2 = weights["W2"], weights["b2"]
        self._tW1, self._tb1 = self._W1.copy(), self._b1.copy()
        self._tW2, self._tb2 = self._W2.copy(), self._b2.copy()

    def statistics(self):
        return {"total_updates": self._total_updates, "buffer_size": len(self._replay)}


# ── TorchDQN Adapter ─────────────────────────────────────────────────

class TorchDQNLearner(Learner):
    """
    Wraps our TorchDQN to implement the Learner interface.

    This allows the MetaController and RLRegistry to use our
    existing PyTorch DQN as one of the algorithm options.
    """

    def __init__(self, n_features=84, n_actions=18, **kwargs):
        try:
            from brain.learning.torch_dqn import TorchDQN
            self._dqn = TorchDQN(
                n_features=n_features,
                n_actions=n_actions,
                **{k: v for k, v in kwargs.items() if k in (
                    'hidden_sizes', 'lr', 'gamma', 'tau',
                    'buffer_size', 'batch_size', 'grad_clip',
                    'epsilon_start', 'epsilon_end', 'epsilon_decay_steps',
                    'dropout', 'use_cnn', 'device',
                )},
            )
        except ImportError:
            raise ImportError("TorchDQN requires PyTorch")

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        action, _ = self._dqn.select_action(state, explore=epsilon > 0)
        return action

    def update(self, state, action, reward, next_state, done):
        self._dqn.store_transition(state, action, reward, next_state, done)
        return self._dqn.train_step()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self._dqn.forward(state)

    def save_weights(self):
        return {"state_dict": self._dqn.online_net.state_dict()}

    def load_weights(self, weights):
        if "state_dict" in weights:
            self._dqn.online_net.load_state_dict(weights["state_dict"])

    def statistics(self):
        return self._dqn.stats()


def _create_torch_dqn(**kwargs) -> Learner:
    """Create the TorchDQN learner."""
    return TorchDQNLearner(**kwargs)

