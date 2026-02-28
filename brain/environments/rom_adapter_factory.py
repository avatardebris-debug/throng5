"""
rom_adapter_factory.py — Auto-detect and create environment adapters.

Given any gym/ale/retro environment, automatically detects:
  - Action space type and size (discrete/continuous)
  - Observation space shape and type (RAM, pixel, vector)
  - Reward characteristics (sparse/dense, range)
  - Episode length statistics

Then creates the appropriate adapter wrapping the environment
in the brain's abstract feature interface.

Usage:
    from brain.environments.rom_adapter_factory import ROMAdapterFactory

    factory = ROMAdapterFactory()
    adapter, fingerprint = factory.create("MontezumaRevenge-v5")
    # or
    adapter, fingerprint = factory.create_from_env(env)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


@dataclass
class EnvFingerprint:
    """Auto-detected characteristics of an environment."""
    name: str
    platform: str = "unknown"          # "atari", "nes", "gymnasium", "custom"

    # Action space
    action_space_type: str = "discrete"  # "discrete" or "continuous"
    n_actions: int = 18
    action_meanings: List[str] = field(default_factory=list)

    # Observation space
    obs_type: str = "vector"           # "vector", "ram", "pixels"
    obs_shape: Tuple[int, ...] = (84,)
    obs_dtype: str = "float32"

    # Reward characteristics (measured during probe)
    reward_density: str = "unknown"    # "sparse", "moderate", "dense"
    reward_range: Tuple[float, float] = (0.0, 0.0)
    avg_episode_length: float = 0.0

    # Capabilities
    has_ram: bool = False
    has_lives: bool = False
    is_deterministic: bool = False


class UniversalAdapter:
    """
    Universal environment adapter that works with any obs format.

    Converts any observation into the brain's 84-dim abstract feature vector:
    - RAM (128 bytes) → normalize + project
    - Pixels (210×160×3) → downsample + flatten + project
    - Vector (N dims) → pad/truncate + normalize
    """

    def __init__(
        self,
        fingerprint: EnvFingerprint,
        target_dim: int = 84,
    ):
        self.fingerprint = fingerprint
        self.target_dim = target_dim

        self._obs_buffer: Optional[np.ndarray] = None
        self._reward_history: List[float] = []
        self._step_count = 0

        # Projection matrix for dimensionality reduction
        rng = np.random.RandomState(42)
        obs_flat_size = int(np.prod(fingerprint.obs_shape))
        if obs_flat_size > target_dim:
            self._projection = rng.randn(obs_flat_size, target_dim).astype(np.float32)
            self._projection /= np.sqrt(obs_flat_size)  # Xavier-like init
        else:
            self._projection = None

        # Running stats for normalization
        self._mean = np.zeros(target_dim, dtype=np.float32)
        self._var = np.ones(target_dim, dtype=np.float32)
        self._n_obs = 0

    def observe(self, obs: Any) -> None:
        """Store the latest observation."""
        self._obs_buffer = obs
        self._step_count += 1

    def make_features(self, action: int = 0) -> np.ndarray:
        """Convert current observation into 84-dim abstract features."""
        if self._obs_buffer is None:
            return np.zeros(self.target_dim, dtype=np.float32)

        raw = np.asarray(self._obs_buffer, dtype=np.float32).flatten()

        if self._projection is not None:
            # Project high-dim obs to target_dim
            if len(raw) > self._projection.shape[0]:
                raw = raw[:self._projection.shape[0]]
            elif len(raw) < self._projection.shape[0]:
                raw = np.pad(raw, (0, self._projection.shape[0] - len(raw)))
            features = raw @ self._projection
        else:
            # Pad/truncate to target_dim
            if len(raw) < self.target_dim:
                features = np.pad(raw, (0, self.target_dim - len(raw)))
            else:
                features = raw[:self.target_dim]

        # Running normalization
        self._update_stats(features)
        features = (features - self._mean) / (np.sqrt(self._var) + 1e-8)

        return features.astype(np.float32)

    def _update_stats(self, features: np.ndarray) -> None:
        """Update running mean and variance for normalization."""
        self._n_obs += 1
        alpha = max(0.01, 1.0 / self._n_obs)
        self._mean = (1 - alpha) * self._mean + alpha * features
        self._var = (1 - alpha) * self._var + alpha * (features - self._mean) ** 2


class ROMAdapterFactory:
    """
    Factory that auto-detects environment properties and creates adapters.

    Works with gymnasium, ale-py, and gym-retro environments.
    """

    def __init__(self, target_dim: int = 84, probe_steps: int = 100):
        self._target_dim = target_dim
        self._probe_steps = probe_steps

    def create_from_env(self, env, name: str = "unknown") -> Tuple[UniversalAdapter, EnvFingerprint]:
        """
        Create an adapter from an existing environment.

        Auto-detects action space, observation space, and reward characteristics.
        """
        fingerprint = self._detect_fingerprint(env, name)
        adapter = UniversalAdapter(fingerprint, target_dim=self._target_dim)
        return adapter, fingerprint

    def create(self, env_id: str, **kwargs) -> Tuple[UniversalAdapter, EnvFingerprint]:
        """
        Create an adapter from an environment ID string.

        Tries gymnasium first, then ale-py, then gym-retro.
        """
        env = self._make_env(env_id, **kwargs)
        if env is None:
            # Fallback: create a default fingerprint
            fingerprint = EnvFingerprint(name=env_id)
            return UniversalAdapter(fingerprint), fingerprint

        adapter, fingerprint = self.create_from_env(env, env_id)
        env.close()
        return adapter, fingerprint

    def _detect_fingerprint(self, env, name: str) -> EnvFingerprint:
        """Auto-detect all environment characteristics."""
        fp = EnvFingerprint(name=name)

        # ── Action space ──────────────────────────────────────────────
        action_space = getattr(env, "action_space", None)
        if action_space is not None:
            space_type = type(action_space).__name__

            if "Discrete" in space_type:
                fp.action_space_type = "discrete"
                fp.n_actions = int(action_space.n)
            elif "Box" in space_type:
                fp.action_space_type = "continuous"
                fp.n_actions = int(np.prod(action_space.shape))
            elif "MultiBinary" in space_type:
                fp.action_space_type = "discrete"
                fp.n_actions = int(action_space.n)
            elif "MultiDiscrete" in space_type:
                fp.action_space_type = "discrete"
                fp.n_actions = int(np.sum(action_space.nvec))

            # Try to get action meanings (Atari-specific)
            if hasattr(env, "get_action_meanings"):
                try:
                    fp.action_meanings = env.get_action_meanings()
                except Exception:
                    pass

        # ── Observation space ─────────────────────────────────────────
        obs_space = getattr(env, "observation_space", None)
        if obs_space is not None:
            fp.obs_shape = tuple(obs_space.shape)
            fp.obs_dtype = str(obs_space.dtype)

            # Classify observation type
            if len(fp.obs_shape) == 1 and fp.obs_shape[0] == 128:
                fp.obs_type = "ram"
                fp.has_ram = True
            elif len(fp.obs_shape) == 3:
                fp.obs_type = "pixels"
            else:
                fp.obs_type = "vector"

        # ── Platform detection ────────────────────────────────────────
        env_class = type(env).__name__.lower()
        spec_id = getattr(getattr(env, "spec", None), "id", "").lower()

        if "atari" in env_class or "ale" in env_class or "atari" in spec_id:
            fp.platform = "atari"
        elif "retro" in env_class:
            fp.platform = "nes"
        elif "mujoco" in env_class:
            fp.platform = "mujoco"
        else:
            fp.platform = "gymnasium"

        # ── Lives detection ───────────────────────────────────────────
        if hasattr(env, "ale"):
            fp.has_lives = True
        elif hasattr(env, "lives"):
            fp.has_lives = True

        # ── Probe for reward characteristics ──────────────────────────
        fp = self._probe_rewards(env, fp)

        return fp

    def _probe_rewards(self, env, fp: EnvFingerprint) -> EnvFingerprint:
        """Run a short random probe to measure reward characteristics."""
        try:
            obs_result = env.reset()
            # Handle both old (obs) and new (obs, info) reset API
            if isinstance(obs_result, tuple):
                obs = obs_result[0]
            else:
                obs = obs_result

            rewards = []
            steps = 0
            episode_lengths = []
            ep_steps = 0

            for _ in range(self._probe_steps):
                action = env.action_space.sample()
                step_result = env.step(action)

                # Handle both old (obs, r, done, info) and new (obs, r, term, trunc, info) APIs
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    break

                rewards.append(float(reward))
                steps += 1
                ep_steps += 1

                if done:
                    episode_lengths.append(ep_steps)
                    ep_steps = 0
                    obs_result = env.reset()
                    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result

            if rewards:
                nonzero = sum(1 for r in rewards if abs(r) > 0.001)
                ratio = nonzero / len(rewards)
                if ratio < 0.05:
                    fp.reward_density = "sparse"
                elif ratio < 0.3:
                    fp.reward_density = "moderate"
                else:
                    fp.reward_density = "dense"
                fp.reward_range = (min(rewards), max(rewards))

            if episode_lengths:
                fp.avg_episode_length = float(np.mean(episode_lengths))

        except Exception:
            pass  # Probe is best-effort

        return fp

    def _make_env(self, env_id: str, **kwargs):
        """Try to create an environment from an ID string."""
        # Try gymnasium first
        try:
            import gymnasium as gym
            return gym.make(env_id, **kwargs)
        except Exception:
            pass

        # Try old gym
        try:
            import gym
            return gym.make(env_id, **kwargs)
        except Exception:
            pass

        # Try ale-py directly
        try:
            import ale_py
            ale = ale_py.ALEInterface()
            ale.loadROM(env_id)
            # Can't easily wrap this, skip
        except Exception:
            pass

        return None
